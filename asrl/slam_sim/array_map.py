import os
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
    def __init__(self, arr: str, resolution=None, verbose=False):
        """
        ArrayMap class to build a 2D map with open3d geometry from a 2D numpy idealized
        occupancy grid.

        Parameters
        ----------
        arr : str
            The array to be converted into a map. It can be a string representing the file path
            or a string representation of the array itself. The first line should contain
            the characters representing occupied and free space, respectively. Then, the map lines follow.
        resolution : float, optional
            The scale of the map in meters, by default 1.0. The size of each cell in the input grid in meters.
        """
        
        if os.path.isfile(arr):
            with open(arr, 'r') as f:
                lines = f.readlines()
                resolution = float(lines[0].strip())
                lines = lines[1:]
        else:
            lines = arr.strip().splitlines()
            
        arr = self._map_from_lines(lines)
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
        
        self.resolution = resolution
        self.height_px, self.width_px = self._walls.shape
        self.height_m, self.width_m = self.height_px * self.resolution, self.width_px * self.resolution
        
        self._indxer = ArrayIndexer(1.0, self.height_px, self.width_px) # TODO: add resolution to this
        
        if verbose:
            print(f"Walls shape: {self._walls.shape}, Free space shape: {self._free_space.shape}")
            print(self._walls)
        
        self._free_space_indices_ij = np.argwhere(self._free_space == 1)
        self.free_area_m2 = len(self._free_space_indices_ij) * (self.resolution ** 2)
        
        # Create the open3d geometries for visualization and raycasting
        self._wall_o3d_geometries = self._to_o3d_geometry(self._walls)
        
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        for cube in self._wall_o3d_geometries:
            self._raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cube))
    
    def _map_from_lines(self, lines: list[str]):
        # Parse character legend
        occupied_char, free_char = lines[0].strip()
        map_lines = lines[1:]

        # Convert map lines into binary grid
        grid = np.array([
            [0 if char == occupied_char else 1 for char in line.strip()]
            for line in map_lines
        ], dtype=np.uint8)

        return grid

    # def _to_o3d_geometry(self, walls: NDArray):        
    #     def create_cube(xy_coord):
    #         x, y = xy_coord - 0.5
    #         cube = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
            
    #         cube.compute_vertex_normals()
    #         cube.translate([x, y, 0])
    #         cube.paint_uniform_color([1, 0, 0])
    #         return cube
        
    #     wall_point_xy_m = self._indxer.ij_to_xy_m(np.argwhere(walls == 1))
    #     cube_geometries = list(map(create_cube, wall_point_xy_m))
        
    #     return cube_geometries
    
    def _to_o3d_geometry(self, walls: NDArray):
        inset = 0.05
        cube_size = 1.0

        def create_cube(coord_ij):
            i, j = coord_ij
            x, y = self._indxer.ij_to_xy_m(np.array([i, j]))

            # Start with full-size cube
            scale_x = cube_size
            scale_y = cube_size
            offset_x = 0.0
            offset_y = 0.0

            # Offsets for each direction
            neighbors = {
                'N': (i - 1, j),
                'S': (i + 1, j),
                'W': (i, j - 1),
                'E': (i, j + 1),
            }

            for direction, (ni, nj) in neighbors.items():
                if 0 <= ni < self.height_px and 0 <= nj < self.width_px:
                    if self._free_space[ni, nj]:  # wall faces free space
                        if direction == 'W':
                            offset_x += inset / 2
                            scale_x -= inset
                        elif direction == 'E':
                            offset_x -= inset / 2
                            scale_x -= inset
                        elif direction == 'S':
                            offset_y += inset / 2
                            scale_y -= inset
                        elif direction == 'N':
                            offset_y -= inset / 2
                            scale_y -= inset
                            
            # Construct cube with adjusted size and position
            cube = o3d.geometry.TriangleMesh.create_box(width=scale_x, height=scale_y, depth=1.0)
            cube.compute_vertex_normals()

            # Translate cube to the right place in meters
            cube.translate([x - scale_x / 2 + offset_x, y - scale_y / 2 + offset_y, 0])
            cube.paint_uniform_color([1, 0, 0])
            return cube

        wall_indices_ij = np.argwhere(walls == 1)
        cube_geometries = list(map(create_cube, wall_indices_ij))
        return cube_geometries
    
    def sample_free_space(self, output_type='xy_m'):
        assert output_type in ['ij', 'xy_m', 'xy_r'], f"Invalid output type: {output_type}"
        
        index = np.random.randint(self._free_space_indices_ij.shape[0])
        free_space_point_ij = self._free_space_indices_ij[index]
        
        if output_type == 'ij': return free_space_point_ij
        if output_type == 'xy_m': return self._indxer.ij_to_xy_m(free_space_point_ij)
        if output_type == 'xy_r': return self._indxer.ij_to_xy_r(free_space_point_ij)
    
    def max_travel_distance_along_ray(self, position: NDArray, angle_W: NDArray) -> float:
        """
        Calculate the maximum travel distance along a ray from a given pose in the map.

        Parameters
        ----------
        position : NDArray
            The position of the ray in the map (x, y) in meters.
        angle_W : NDArray
            The direction of the ray in radians with respect to the world frame.

        Returns
        -------
        float
            The maximum travel distance along the ray before hitting an obstacle.
        """
        raycast_vectors = np.zeros((1, 6), dtype=np.float32)
        raycast_vectors[0, 3] = np.cos(angle_W)
        raycast_vectors[0, 4] = np.sin(angle_W)
        raycast_vectors[0, :2] = position
        
        out = self._raycasting_scene.cast_rays(raycast_vectors)
        t_hit = out['t_hit'].numpy()
        
        return t_hit[0]
    
    def raycast_in_map(self,
                       poses: NDArray,
                       r_max_m: float = np.inf,
                       angle_range_deg: float = [-180, 180],
                       horizontal_resolution_deg: float = 2.0,
                       range_noise_m: float = 0.01):
        is_batched = poses.ndim == 2
        if not is_batched:
            poses = poses[None, :]
    
        directions = np.deg2rad(np.arange(*angle_range_deg, horizontal_resolution_deg, dtype=np.float32))
        n_rays = len(directions)
        B = poses.shape[0]

        raycast_vectors = np.zeros((B * n_rays, 6), dtype=np.float32)

        dir_cos = np.cos(directions)
        dir_sin = np.sin(directions)

        raycast_vectors[:, 3] = np.tile(dir_cos, B)
        raycast_vectors[:, 4] = np.tile(dir_sin, B)
        raycast_vectors[:, :2] = np.repeat(poses[:, :2], n_rays, axis=0)
        raycast_vectors[:, 2] = 0.5 * self.resolution
        
        # Perform raycasting
        out = self._raycasting_scene.cast_rays(raycast_vectors)
        t_hit = out['t_hit'].numpy()
        
        sigma = range_noise_m + 0.001 * t_hit  # base noise + 1mm per meter
        noise = np.random.normal(loc=0.0, scale=sigma)
        t_hit += noise
        
        # Rotate points into body frame
        B_R_W_array = np.array([spatialmath.base.rot2(pose[2]).T for pose in poses])
        points_b_W = np.reshape(t_hit[:, None] * raycast_vectors[:, 3:5], (B, n_rays, 2))
        points_b_B = points_b_W @ B_R_W_array.transpose(0, 2, 1)
        
        mask_valid = np.reshape((t_hit <= r_max_m) & np.isfinite(t_hit), (B, n_rays))
        result = [
            points_b_B[i, ...][mask_valid[i]] for i in range(B)
        ]
        
        if not is_batched:
            return result[0]

        return result
    

if __name__ == "__main__":
    m = ArrayMap('/home/dev/workspace/asrl/maps/floorplan1.txt', resolution=1)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    grid = create_grid_xy(
        x_range=(0, m.width_px),
        y_range=(0, m.height_px),
        step=1.0
    )
    o3d.visualization.draw_geometries(m._wall_o3d_geometries + [coordinate_frame, grid])
    
    pose = np.array([0.5 + 1.5, 0.5 + 2.5, np.deg2rad(10)])
    points = m.raycast_in_map(
        pose,
        range_noise_std_m=0.
    )
    
    fig, ax = plt.subplots(1,1, figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], c='r', s=1, label='Hit Points')
    ax.scatter([0], [0], c='b', s=20, marker='x', label='Robot')
    ax.set_aspect('equal')
    plt.show()