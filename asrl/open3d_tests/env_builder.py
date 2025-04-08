import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, label, find_objects
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
from spatialmath.base import angle_wrap
from spatialmath import SO3
from time import sleep
from scipy.ndimage import zoom
from scipy.ndimage import distance_transform_edt


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
    def __init__(self, arr: np.ndarray, scale=1.0):
        arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
        # arr = zoom(arr, scale, order=0)
        
        structure = np.ones((3, 3), dtype=int)
        dilated = binary_dilation(arr, structure=structure).astype(int)
        walls = dilated - arr
        
        h, w = walls.shape
        
        crop_min_i, crop_max_i = h, 0
        for i in range(h):
            if np.any(walls[i]):
                crop_min_i = min(crop_min_i, i)
                crop_max_i = max(crop_max_i, i)

        crop_min_j, crop_max_j = w, 0
        for j in range(w):
            if np.any(walls[:, j]):
                crop_min_j = min(crop_min_j, j)
                crop_max_j = max(crop_max_j, j)
                
        self.env_bounds = (crop_min_i, crop_max_i, crop_min_j, crop_max_j)
        self.walls = walls[crop_min_i:crop_max_i+1, crop_min_j:crop_max_j+1]
        self.free_space = arr[crop_min_i:crop_max_i+1, crop_min_j:crop_max_j+1]
        # self.distance = distance_transform_edt(self.free_space)
        self.costmap = self._build_costmap(self.free_space)
        
        print(self.walls)
        
        self.free_space_indices = np.argwhere(self.free_space == 1)
        self.scale = scale
    
    def _build_costmap(self, free_space):
        distances = distance_transform_edt(free_space)
        return np.exp(-1/4 * distances)
    
    def coord_to_xy(self, coord, shape):
        if len(coord.shape) == 1:
            coord = np.expand_dims(coord, axis=0)
            
        i, j = coord.T
        out = np.array([j, shape[0] - i - 1]) + 0.5
        
        return np.squeeze(out.T)
    
    def xy_to_coord(self, xy, shape):
        if len(xy.shape) == 1:
            xy = np.expand_dims(xy, axis=0)
        
        x, y = xy.T - 0.5
        
        out = np.floor(np.array([shape[0] - y - 1, x])).astype(int)
        
        return np.squeeze(out.T)
    
    def sample_free_space(self, n: int = 1, kind='xy', replace=False):
        assert kind in ['xy', 'coord'], f"Invalid kind: {kind}"
        
        inds = np.random.choice(
            np.arange(self.free_space_indices.shape[0]),
            size=n,
            replace=replace
        )
        
        if kind == 'coord':
            out = self.free_space_indices[inds]
        else:
            x, y = self.coord_to_xy(self.free_space_indices[inds])
            out = np.column_stack((x, y))
            
        if n == 1:
            return out[0]
        return out
    
    def to_o3d_geometry(self):
        walls = []
        h, w = self.walls.shape
        
        for i in range(h):
            for j in range(w):
                if self.walls[i, j]:
                    wall_x, wall_y = j, h - i - 1
                    
                    cube = o3d.geometry.TriangleMesh.create_box(1, 1, 3)
                    cube.compute_vertex_normals()
                    cube.translate([wall_x, wall_y, 0])
                    cube.paint_uniform_color([1, 0, 0])
                    
                    walls.append(cube)
        
        floor = []
        for i in range(h):
            for j in range(w):
                if self.free_space[i, j]:
                    floor_x, floor_y = j, h - i - 1
                    cube_height = 0.1
                    cube = o3d.geometry.TriangleMesh.create_box(1, 1, cube_height)
                    cube.compute_vertex_normals()
                    cube.translate([floor_x, floor_y, -cube_height])
                    cube.paint_uniform_color([0, 1, 0])
                    
                    floor.append(cube)
        
        return walls + floor
    
    def plan_path_astar(self, start: np.ndarray, goal: np.ndarray, resolution=10):
        free_space = zoom(self.free_space, resolution, order=0)
        costmap = self._build_costmap(free_space)
        
        # start, goal are xy coords, so convert them to coords
        start = tuple( self.xy_to_coord(start * resolution, free_space.shape) )
        goal = tuple( self.xy_to_coord(goal * resolution, free_space.shape) )
                
        def heuristic(a, b):
            return 50 * resolution * costmap[a] + np.linalg.norm(np.array(a) - np.array(b))  # Euclidean

        rows, cols = free_space.shape
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))  # (f, g, node)

        came_from = {}
        g_score = {start: 0}
        visited = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 4-connected
        # For 8-connected, add: (-1, -1), (-1, 1), (1, -1), (1, 1)

        path = None
                
        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                break

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue
                if free_space[neighbor] == 0:  # obstacle
                    continue

                tentative_g = current_g + np.linalg.norm([dx, dy])
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current

        if path is not None:
            path = self.coord_to_xy(np.asarray(path), free_space.shape)
            return path, free_space
            # path = path / resolution
            
            # return path
        
        return None  # No path found


def collect_body_lidar_scans(poses, scene, r_max=np.inf):
    directions = np.deg2rad(np.arange(-180, 180, 2, dtype=np.float32))
    directions_vectors = np.column_stack(
        (np.cos(directions), np.sin(directions), np.zeros_like(directions))
    )
    
    scan_points = [] #np.empty((len(poses), len(directions), 3), dtype=np.float32)
    
    for i in range(len(poses)):
        raycast_origin = 0.5 * np.ones(3, dtype=np.float32)
        raycast_origin[:2] = poses[i, :2]
        
        raycast_vectors = np.column_stack(
            (raycast_origin[None, :] * np.ones_like(directions_vectors), directions_vectors)
        )
        
        ans = scene.cast_rays(raycast_vectors)
        t_hit = ans['t_hit'].numpy()
        t_hit_noise = np.random.normal(0, 0.0254, size=t_hit.shape)
        t_hit += t_hit_noise

        hit = t_hit < r_max

        theta = poses[i, 2]
        W_R_B = SO3.Rz(theta).A
        
        points_W = raycast_vectors[hit][:, 3: ] * t_hit[hit].reshape((-1, 1))
        points_B = points_W @ W_R_B
        
        scan_points.append(points_B)

    return scan_points


def generate_sphere_directions(h_res_deg=1.0, v_res_deg=2.0,
                               v_fov_deg_tot=30):
    # Horizontal: azimuth angles (yaw), full circle
    azimuths = np.arange(-180, 180, h_res_deg, dtype=np.float32)
    azimuths = np.deg2rad(azimuths)

    # Vertical: elevation angles (pitch), from -90 (down) to +90 (up)
    elevations = np.arange(-v_fov_deg_tot/2, v_fov_deg_tot/2 + v_res_deg, v_res_deg, dtype=np.float32)
    elevations = np.deg2rad(elevations)

    # Meshgrid for all (azimuth, elevation) pairs
    azim_grid, elev_grid = np.meshgrid(azimuths, elevations)

    # Spherical to Cartesian unit vectors
    x = np.cos(elev_grid) * np.cos(azim_grid)
    y = np.cos(elev_grid) * np.sin(azim_grid)
    z = np.sin(elev_grid)

    directions = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # shape: (N, 3)

    return directions.astype(np.float32)


def add_lidar_noise(points, range_std=0.03, angular_jitter_deg=0.1, dropout_prob=0.01):
    """
    Add realistic noise to ground-truth LiDAR points.
    
    Parameters:
        points (N, 3) array of (x, y, z) lidar points
        range_std: standard deviation of range noise (in meters)
        angular_jitter_deg: std deviation of angular jitter (degrees)
        dropout_prob: probability that a point is dropped
        
    Returns:
        Noisy (N, 3) point array with NaNs for dropped points (optional)
    """
    if points.shape[1] != 3:
        raise ValueError("Points must be (N, 3) array")

    # Convert to spherical (r, azimuth, elevation)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)

    # Add Gaussian noise to range
    r_noisy = r + np.random.normal(0, range_std, size=r.shape)

    # Add angular jitter
    azimuth += np.deg2rad(np.random.normal(0, angular_jitter_deg, size=r.shape))
    elevation += np.deg2rad(np.random.normal(0, angular_jitter_deg, size=r.shape))

    # Reconstruct noisy points
    x_noisy = r_noisy * np.cos(elevation) * np.cos(azimuth)
    y_noisy = r_noisy * np.cos(elevation) * np.sin(azimuth)
    z_noisy = r_noisy * np.sin(elevation)
    noisy_points = np.stack([x_noisy, y_noisy, z_noisy], axis=1)

    # Randomly drop some points
    mask = np.random.rand(len(points)) >= dropout_prob
    # noisy_points[mask] = np.nan  # or you could remove them entirely

    return noisy_points[mask]


if __name__ == "__main__":
    # vectors = generate_sphere_directions(10, 10, v_fov_deg_tot=90)
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.plot(vectors[:, 0], vectors[:, 1], vectors[:, 2], 'o')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_aspect('equal')
    # plt.show()
    
    og = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ])
    
    m = ArrayMap(og, scale=1)
    
    # start = m.coord_to_xy(np.array([1, 1]), m.free_space.shape)
    # goal = m.coord_to_xy(np.array([6, 2]), m.free_space.shape)
    
    # print(f"Start xy: {start}", start.shape)
    # print(f"Goal xy: {goal}", goal.shape)
    
    # path = m.plan_path_astar(start, goal)

    env_cubes = m.to_o3d_geometry()

    # Create a coordinate frame to better visualize orientation
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    walls_ymax, walls_xmax = m.walls.shape
    grid_xy = create_grid_xy(x_range=(0, walls_xmax), y_range=(0, walls_ymax), step=1.0)

    # Visualize the cube (and coordinate frame) in a scene
    o3d.visualization.draw_geometries(env_cubes + [coordinate_frame, grid_xy])



    # Get 3D point cloud
    scene = o3d.t.geometry.RaycastingScene()
    
    for cube in env_cubes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cube))

    directions = generate_sphere_directions(1, 2, v_fov_deg_tot=30)
    n_rays = directions.shape[0]
    # raycast_origin = 1/2 * np.ones(3, dtype=np.float32)
    raycast_origins = np.tile(np.array([2.5, 5.5, 0.5])[None, :], (n_rays, 1)).astype(np.float32)
    
    raycast_vectors = np.column_stack(
        [raycast_origins, directions]
    )
    
    print(directions.shape)
    print(raycast_origins.shape)
    print(raycast_vectors.shape)
    
    max_range = np.inf
    ans = scene.cast_rays(raycast_vectors)
    t_hit = ans['t_hit'].numpy()
    hit = t_hit < max_range

    hit_points = raycast_vectors[hit][:, :3] + raycast_vectors[hit][:, 3: ] * t_hit[hit].reshape((-1, 1))
    hit_points = add_lidar_noise(hit_points, range_std=0.03, angular_jitter_deg=0.05, dropout_prob=0.01)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], 'o', markersize=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    plt.show()
    
    # pts = m.sample_free_space(2, kind='coord', replace=False)
    # path = m.plan_path_astar(tuple(pts[0]), tuple(pts[1]))
    # path = m.coord_to_xy(path)
    
    # controller = DiffDrivePID(kp_v=0.5, kp_w=2.0, max_v=1.0, max_w=np.pi/4)
    # traversed_path = np.asarray(travel_along_path(path, 0.1, controller, 0.1))

    # # Build o3d scene
    # walls = m.to_o3d_geometry()
    # scene = o3d.t.geometry.RaycastingScene()
    
    # for cube in walls:
    #     scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cube))
        
    # scans = collect_body_lidar_scans(traversed_path, scene)
    
    # og = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 0, 1, 1, 1],
    #     [0, 1, 1, 1, 0, 1, 1, 0],
    #     [0, 0, 1, 1, 1, 1, 1, 1],
    #     [0, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 0, 0],
    # ])
    
    # m = ArrayMap(og, scale=1.)
    # print(m.walls)
    # print("free space:")
    # print(m.free_space)
    # pt = m.sample_free_space(1, kind='xy')
    # # pt = np.array([1.5, 4.5])
    # print(pt)
    
    # pt1 = m.sample_free_space(1, kind='coord')
    # print(pt1)
    # pt2 = m.sample_free_space(1, kind='coord')
    # print(pt2)
    # out = m.plan_path_astar(tuple(pt1), tuple(pt2))
    # print(out)
    # m.visualize_path(out)
    
    # exit(1)
    
    # walls = m.to_o3d_geometry()

    # # Create a coordinate frame to better visualize orientation
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # pt_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    # pt_marker.translate([pt[0], pt[1], m.scale/2])
    
    # grid_xy = create_grid_xy(x_range=(-5, 5), y_range=(-5, 5), step=1.0)

    # # Visualize the cube (and coordinate frame) in a scene
    # o3d.visualization.draw_geometries(walls + [coordinate_frame, grid_xy, pt_marker])
    
    # scene = o3d.t.geometry.RaycastingScene()
    # for cube in walls:
    #     scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cube))
    
    # directions = np.deg2rad(np.arange(-180, 180, 1, dtype=np.float32))
    # directions_vectors = np.column_stack(
    #     (np.cos(directions), np.sin(directions), np.zeros_like(directions))
    # )
    
    # raycast_origin = m.scale / 2 * np.ones(3, dtype=np.float32)
    # raycast_origin[:2] = pt
    
    # raycast_vectors = np.column_stack(
    #     (raycast_origin[None, :] * np.ones_like(directions_vectors), directions_vectors)
    # )
    
    # max_range = np.inf
    # ans = scene.cast_rays(raycast_vectors)
    # t_hit = ans['t_hit'].numpy()
    # t_hit_noise = np.random.normal(0, 0.0254, size=t_hit.shape)
    # t_hit += t_hit_noise

    # hit = t_hit < max_range

    # points = raycast_vectors[hit][:, :3] + raycast_vectors[hit][:, 3: ] * t_hit[hit].reshape((-1, 1))
    
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # ax.scatter(points[:, 0], points[:, 1], c='r', s=1, label='Hit Points')
    # ax.scatter(pt[0], pt[1], c='b', s=10, marker='x', label='Robot')
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.legend()
    # plt.show()
    
    # n_trials = 1_000
    # start = perf_counter()
    # for _ in range(n_trials):
    #     ans = scene.cast_rays(raycast_vectors)
    # t_elapsed = perf_counter() - start
    # print(f"Avg. time per raycast: {t_elapsed/n_trials:.5f} seconds")

    # t_hit = ans['t_hit'].numpy()