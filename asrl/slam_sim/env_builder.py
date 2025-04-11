import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import spatialmath
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from asrl.ogmapping.og_map import ArrayIndexer


def create_grid_xy(x_range, y_range, step):
    """
    Create a grid on the xy plane (z = 0) with given x and y ranges and grid spacing.
    
    Parameters:
      x_range: Tuple (min, max) for the x-axis.
      y_range: Tuple (min, max) for the y-axis.
      step: Grid spacing.
      
    Returns:
      An Open3D LineSet representing the grid.
    """
    points = []
    lines = []
    
    # Generate x and y values within the given ranges
    x_vals = np.arange(x_range[0], x_range[1] + step, step)
    y_vals = np.arange(y_range[0], y_range[1] + step, step)
    
    # Vertical lines: constant x, varying y
    for x in x_vals:
        start = [x, y_range[0], 0]
        end = [x, y_range[1], 0]
        points.append(start)
        points.append(end)
        lines.append([len(points) - 2, len(points) - 1])
    
    # Horizontal lines: constant y, varying x
    for y in y_vals:
        start = [x_range[0], y, 0]
        end = [x_range[1], y, 0]
        points.append(start)
        points.append(end)
        lines.append([len(points) - 2, len(points) - 1])
    
    # Create and color the LineSet for the grid
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    grid.paint_uniform_color([0.8, 0.8, 0.8])  # light gray
    return grid


class ArrayMap:
    def __init__(self, arr: NDArray, resolution=1.0, verbose=False):
        """
        ArrayMap class to build a 2D map with open3d geometry from a 2D numpy idealized
        occupancy grid.

        Parameters
        ----------
        arr : NDArray
            2D numpy array representing the occupancy grid; 0 for free space, 1 elsewhere.
        scale : float, optional
            The scale of the map in meters, by default 1.0. The size of each cell in the input grid in meters.
        """
        arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)

        structure = np.ones((3, 3), dtype=int)
        dilated = binary_dilation(arr, structure=structure).astype(int)
        walls = dilated - arr
        
        rows = np.any(walls, axis=1)
        cols = np.any(walls, axis=0)
        row_start, row_end = np.where(rows)[0][[0, -1]]
        col_start, col_end = np.where(cols)[0][[0, -1]]
                
        self._walls = walls[row_start:row_end+1, col_start:col_end+1]
        self._free_space = arr[row_start:row_end+1, col_start:col_end+1]
        self._indxer = ArrayIndexer(1.0, *self._walls.shape) # TODO: add resolution to this
        
        if verbose:
            print(f"Walls shape: {self._walls.shape}, Free space shape: {self._free_space.shape}")
            print(self._walls)
        
        self._free_space_indices_ij = np.argwhere(self._free_space == 1)
        
        # Create the open3d geometries for visualization and raycasting
        self._wall_o3d_geometries = self._to_o3d_geometry(self._walls)
        
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        for cube in self._wall_o3d_geometries:
            self._raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cube))
    
    def _to_o3d_geometry(self, walls: NDArray):        
        def create_cube(xy_coord):
            x, y = xy_coord - 0.5
            cube = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
            
            cube.compute_vertex_normals()
            cube.translate([x, y, 0])
            cube.paint_uniform_color([1, 0, 0])
            return cube
        
        wall_point_xy_m = self._indxer.ij_to_xy_m(np.argwhere(walls == 1))
        cube_geometries = list(map(create_cube, wall_point_xy_m))
        
        return cube_geometries
    
    def max_travel_distance_along_ray(self, pose: NDArray, ray_direction: NDArray) -> float:
        """
        Calculate the maximum travel distance along a ray from a given pose in the map.

        Parameters
        ----------
        pose : NDArray
            The pose of the robot in the map.
        ray_direction : NDArray
            The direction of the ray.

        Returns
        -------
        float
            The maximum travel distance along the ray before hitting an obstacle.
        """
        raycast_vectors = np.zeros((1, 6), dtype=np.float32)
        raycast_vectors[0, 3:5] = ray_direction / np.linalg.norm(ray_direction)
        raycast_vectors[0, :2] = pose[:2]
        
        out = self._raycasting_scene.cast_rays(raycast_vectors)
        t_hit = out['t_hit'].numpy()
        
        return t_hit[0]
    
    def raycast_in_map(self,
                       pose: NDArray,
                       r_max: float = np.inf,
                       angle_range_deg: float = [-180, 180],
                       horizontal_resolution_deg: float = 2.0,
                       range_noise_std_m: float = 0.0254):
        # Build the array of raycast vectors where each row is [x, y, z, u_x, u_y, u_z]
        # where (x, y, z) is the origin of the ray and (u_x, u_y, u_z) is the direction of the ray.
        directions = np.deg2rad(np.arange(*angle_range_deg, horizontal_resolution_deg, dtype=np.float32))
        n_rays = directions.shape[0]

        raycast_vectors = np.zeros((n_rays, 6), dtype=np.float32)

        raycast_vectors[:, [3, 4]] = np.column_stack([np.cos(directions), np.sin(directions)])
        raycast_vectors[:, [0, 1]] = pose[:2]
        raycast_vectors[:, 2] = 0.5 * self._indxer.resolution
        
        # Perform raycasting
        out = self._raycasting_scene.cast_rays(raycast_vectors)
        t_hit = out['t_hit'].numpy()
        valid_rays_mask = (t_hit <= r_max) & np.isfinite(t_hit)
        
        t_hit[valid_rays_mask] += np.random.normal(0, range_noise_std_m, size=t_hit[valid_rays_mask].shape)
        
        # Convert the raycast vectors to points in body frame
        W_R_B = spatialmath.base.rot2(pose[2])
        B_R_W = W_R_B.T
        
        points_b_W = t_hit[valid_rays_mask, None] * raycast_vectors[valid_rays_mask, 3:5]
        points_b_B = points_b_W @ B_R_W.T
        
        return points_b_B
    

if __name__ == "__main__":
    og1 = np.array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,1,1,1,0,1,0],
        [0,1,1,0,1,1,1,0,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,1,1,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,1,0,0,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0],
    ])
    
    
    og3 = np.array([
        [0,0,0,0,0,0,0,0],
        [0,1,1,0,1,1,1,0],
        [0,1,0,0,1,0,1,0],
        [0,1,0,1,1,0,1,0],
        [0,1,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0],
    ])


    m = ArrayMap(og1, resolution=1)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries(m._wall_o3d_geometries + [coordinate_frame])
    
    pose = np.array([1.5, 6.5, np.deg2rad(90)])
    points = m.raycast_in_map(
        pose
    )
    
    fig, ax = plt.subplots(1,1, figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], c='r', s=1, label='Hit Points')
    ax.scatter([0], [0], c='b', s=10, marker='x', label='Robot')
    ax.set_aspect('equal')
    plt.show()