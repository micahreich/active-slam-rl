import numpy as np
from asrl.ogmapping.bresenham_nd import bresenhamline


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
        
        return xy_r
    
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
        
        return ij

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
    
    def process_scan_2(self, pose: np.ndarray, scan_points: np.ndarray) -> None:
        pass
    
    def process_scan(self, pose: np.ndarray, scan_points: np.ndarray) -> None:
        """
        Process a single scan and update the occupancy grid.
        
        Args:
            pose (np.ndarray): The robot's pose (x, y, theta).
            scan_points (np.ndarray): The (x, y) points from the laser scan.
        """
        n_scan_points = scan_points.shape[0]
        W_R_B = SO2(pose[2])
        scan_points_W = scan_points @ W_R_B.T #+ pose[:2]
        
        scan_points_xy_r = np.floor(scan_points_W / self.resolution).astype(np.int32)
        origin = np.zeros_like(scan_points_xy_r)
        
        # scan_point_distances = np.rint(np.linalg.norm(scan_points_xy_r, axis=-1)).astype(np.int32)
    
        bresenham_points, max_iter = bresenhamline(origin, scan_points_xy_r, max_iter=-1)
        # bresenham_points_distances = np.rint(np.linalg.norm(bresenham_points, axis=-1)).astype(np.int32)
        
        bresenham_points = bresenham_points.reshape((n_scan_points, max_iter, 2))
        bresenham_points_distances = np.linalg.norm(bresenham_points, axis=-1)
        scan_points_distances = np.linalg.norm(scan_points_xy_r, axis=-1)[:, np.newaxis]
        
        mask_pre = (bresenham_points_distances < scan_points_distances).flatten()
        mask_hit = (np.isclose(bresenham_points_distances, scan_points_distances)).flatten()

        pose_xy_r = self.indexer.xy_m_to_xy_r(pose[:2])
        pose_xy_i, pose_xy_j = self.indexer.xy_r_to_ij(pose_xy_r)
        
        bresenham_points = bresenham_points.reshape((-1, 2))
        cell_indices_pre = bresenham_points[mask_pre] + pose_xy_r
        cell_indices_hit = bresenham_points[mask_hit] + pose_xy_r
        
        # # print("bresenham points:")
        # # print(bresenham_points)
        
        # # Each scanline has max_iter points from bresenhamline, but some of them go beyond the scanline's end
        # scanline_intervals_indices_pre = []
        # scanline_intervals_indices_hit = []
        
        # scanline_intervals = []
        # start_idx = 0
        
        # for threshold in scan_point_distances:
        #     mask = bresenham_points_distances[start_idx:start_idx + max_iter] >= threshold
            
        #     if np.any(mask):
        #         rel_idx = np.argmax(mask)  # first True, i.e. ending point of this scanline
        #         idx = start_idx + rel_idx
                
        #         scanline_interval = np.arange(start_idx, idx)
        #         scanline_intervals.append(range(start_idx, idx+1))
                
        #         scanline_intervals_indices_pre.append(scanline_interval)
        #         scanline_intervals_indices_hit.append(idx)
                
        #         start_idx += max_iter  # start next search after this index
        
        # scanline_intervals_indices_pre = np.concatenate(scanline_intervals_indices_pre)

        # pose_xy_r = self.indexer.xy_m_to_xy_r(pose[:2])
        # pose_xy_i, pose_xy_j = self.indexer.xy_r_to_ij(pose_xy_r)

        # # Bresenham points are (x, y) pairs in the map's resolution relative to pose, so first
        # # add the robot's pose to get the actual cell indices in the grid
        # cell_indices_pre = bresenham_points[scanline_intervals_indices_pre] + pose_xy_r
        # cell_indices_hit = bresenham_points[scanline_intervals_indices_hit] + pose_xy_r
        
        # Then convert to (i, j) indices to index into grid
        cell_indices_pre_ij = self.indexer.xy_r_to_ij(cell_indices_pre)
        cell_indices_hit_ij = self.indexer.xy_r_to_ij(cell_indices_hit)
        
        np.add.at(self.grid, (cell_indices_pre_ij[:, 0], cell_indices_pre_ij[:, 1]), self.l_free)
        np.add.at(self.grid, (cell_indices_hit_ij[:, 0], cell_indices_hit_ij[:, 1]), self.l_occ)
        
        # Add free cell log-odds for the cell at the robot's pose
        pose_xy_i, pose_xy_j = self.indexer.xy_r_to_ij(pose_xy_r)
        self.grid[pose_xy_i, pose_xy_j] += self.l_free

if __name__ == "__main__":
    og_mapper = OccupancyGridMapper(resolution=0.5, width_m=4, height_m=4)
    
    # Test the indexing
    xy = np.array([[0.0, 0.0],
                   [1.0, 1.0],
                   [0.5, 3.5],
                   [2.4, 1.2]])
    ij = og_mapper.xy_m_to_ij(xy)
    print(f"xy: \n{xy}")
    print(f"ij: \n{ij}")