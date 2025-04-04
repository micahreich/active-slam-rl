import numpy as np
from asrl.ogmapping.bresenham_nd import bresenhamline
from numba import njit


def SO2(theta: float) -> np.ndarray:
    """
    Returns a 2D rotation matrix for a given angle theta.
    
    Args:
        theta (float): Angle in radians.
        
    Returns:
        np.ndarray: 2x2 rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])


class OccupancyGridIndexer:
    def __init__(self, resolution: float, width_m: int, height_m: int) -> None:
        self.resolution = resolution
        self.width = int(width_m / resolution)
        self.height = int(height_m / resolution)

    def xy_r_to_ij(self, xy: np.ndarray) -> np.ndarray:
        # Convert (x, y) coordinates in the map's resolution to row-column (i, j) indices into the map
        squeeze_back = len(xy.shape) == 1
        if squeeze_back:
            xy = xy[np.newaxis, :]
            
        ij = np.zeros_like(xy, dtype=np.int32)
        ij[:, 0] = self.height - xy[:, 1] - 1
        ij[:, 1] = xy[:, 0]
        
        if squeeze_back:
            return np.squeeze(ij)
        
        return ij
    
    def xy_m_to_xy_r(self, xy: np.ndarray) -> np.ndarray:
        # Convert (x, y) coordinates in meters to (x, y) coordinates in the map's resolution
        squeeze_back = len(xy.shape) == 1
        if squeeze_back:
            xy = xy[np.newaxis, :]
        
        xy_r = np.zeros_like(xy)
        xy_r[:, 0] = np.floor(xy[:, 0] / self.resolution).astype(np.int32)
        xy_r[:, 1] = np.floor(xy[:, 1] / self.resolution).astype(np.int32)
        
        if squeeze_back:
            return np.squeeze(xy_r)
        
        return xy_r.astype(np.int32)
    
    def xy_m_to_ij(self, xy: np.ndarray) -> np.ndarray:
        # Convert (x, y) coordinates in meters to row-column (i, j) indices into the map
        squeeze_back = len(xy.shape) == 1
        if squeeze_back:
            xy = xy[np.newaxis, :]
        
        ij = np.zeros_like(xy, dtype=np.int32)
        ij[:, 0] = self.height - np.floor(xy[:, 1] / self.resolution).astype(np.int32) - 1
        ij[:, 1] = np.floor(xy[:, 0] / self.resolution).astype(np.int32)
        
        if squeeze_back:
            return np.squeeze(ij)
        
        return ij.astype(np.int32)

class OccupancyGridMapper:
    def __init__(self, resolution: float, width_m: int, height_m: int,
                 p_hit: float = 0.9,
                 p_miss: float = 0.1) -> None:
        # Resolution refers to the size of each cell in the grid [m]
        self.resolution = resolution
        
        self.l_occ = np.log(p_hit / (1 - p_hit))  # Log-odds for occupied cell
        self.l_free = np.log(p_miss / (1 - p_miss))  # Log-odds for free cell
        self.l_unknown = np.log(0.5 / (1 - 0.5))  # Log-odds for unknown cell
        
        self.width = int(width_m / resolution)
        self.height = int(height_m / resolution)
        self.grid = np.full((self.height, self.width), self.l_unknown, dtype=np.float32)
        
        self.indexer = OccupancyGridIndexer(resolution, width_m, height_m)
    
    def reset_grid(self) -> None:
        """
        Reset the occupancy grid to unknown state.
        """
        self.grid.fill(self.l_unknown)
    
    def log_odds_map_to_prob_map(self) -> np.ndarray:
        """
        Convert log-odds to probability.
        
        Returns:
            np.ndarray: Probability values.
        """
        return 1.0 / (1.0 + np.exp(-self.grid))
    
    def process_scan(self, pose: np.ndarray, scan_points: np.ndarray) -> None:
        """
        Process a single scan and update the occupancy grid.
        
        Args:
            pose (np.ndarray): The robot's pose (x, y, theta).
            scan_points (np.ndarray): The (x, y) points from the laser scan.
        """
        W_R_B = SO2(pose[2])
        scan_points_W = scan_points @ W_R_B.T #+ pose[:2]
                
        scan_points_xy_r = self.indexer.xy_m_to_xy_r(scan_points_W)
        origin = np.zeros_like(scan_points_xy_r)
        
        rays, max_iter = bresenhamline(origin, scan_points_xy_r, max_iter=-1)
        rays = rays.reshape((-1, max_iter, 2))
                
        # Bresenham computes max_iter points along the line for each scan point,
        # but some of these will go beyond the scan point
        ray_dists = np.linalg.norm(rays, axis=-1)
        scan_dists = np.linalg.norm(scan_points_xy_r, axis=-1)[:, np.newaxis]
        
        mask_pre = (ray_dists < scan_dists).flatten()
        mask_hit = (np.isclose(ray_dists, scan_dists)).flatten()

        pose_xy_r = self.indexer.xy_m_to_xy_r(pose[:2])
        pose_xy_i, pose_xy_j = self.indexer.xy_r_to_ij(pose_xy_r)
        
        rays = rays.reshape((-1, 2))
        cell_indices_free = rays[mask_pre] + pose_xy_r
        cell_indices_hit = rays[mask_hit] + pose_xy_r
        
        # Then convert to (i, j) indices to index into grid
        grid_ij_free = self.indexer.xy_r_to_ij(cell_indices_free)
        grid_ij_hit = self.indexer.xy_r_to_ij(cell_indices_hit)
        
        np.add.at(self.grid, (grid_ij_free[:, 0], grid_ij_free[:, 1]), self.l_free)
        np.add.at(self.grid, (grid_ij_hit[:, 0], grid_ij_hit[:, 1]), self.l_occ)
      
        self.grid[pose_xy_i, pose_xy_j] += self.l_free


if __name__ == "__main__":
    import cProfile
    import pstats
    
    og_mapper = OccupancyGridMapper(resolution=0.5, width_m=4, height_m=4)
    
    # Test the indexing
    xy = np.array([[0.0, 0.0],
                   [1.0, 1.0],
                   [0.5, 3.5],
                   [2.4, 1.2]])
    ij = og_mapper.indexer.xy_m_to_ij(xy)
    print(f"xy: \n{xy}")
    print(f"ij: \n{ij}")
    
    # Do some profiling
    og_mapper = OccupancyGridMapper(resolution=0.1, width_m=10, height_m=10,
                                p_hit=0.9, p_miss=0.2)

    pose = np.array([5, 5, 0])
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    radii = 1.0 + 2.0 * np.cumsum(np.ones_like(angles)) / len(angles)

    cos_theta, sin_theta = np.cos(angles), np.sin(angles)
    xs = radii * cos_theta
    ys = radii * sin_theta
    points = np.stack((xs, ys), axis=-1)

    profiler = cProfile.Profile()
    profiler.enable()
    og_mapper.process_scan(pose, points)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(10)  # top 10 slowest