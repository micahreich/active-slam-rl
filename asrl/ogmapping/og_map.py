import numpy as np
from numpy.typing import NDArray
import spatialmath

from asrl.ogmapping.bresenham_nd import bresenhamline
from numba import njit


@njit
def fast_add_at(grid, ij, value):
    for k in range(ij.shape[0]):
        i, j = ij[k]
        grid[i, j] += value


class ArrayIndexer:
    def __init__(self, resolution: float, height_px: int, width_px: int) -> None:
        """
        Array indexer for converting between different coordinate systems and indexing
        into 2D numpy arrays with (x, y) coordinates or (row, column) indices.

        Parameters
        ----------
        resolution : float
            The size of each cell in the grid in meters.
        width : int
            The width of the grid in cells.
        height : int
            The height of the grid in cells.
        """
        self.resolution = resolution
        self.height_px = height_px
        self.width_px = width_px

    def ij_in_bounds(self, ij: NDArray) -> NDArray:
        # Check if (i, j) indices are within the bounds of the grid
        return (0 <= ij[:, 0]) & (ij[:, 0] < self.height_px) & (0 <= ij[:, 1]) & (ij[:, 1] < self.width_px)
    
    def xy_r_in_bounds(self, xy: NDArray) -> NDArray:
        # Check if (x, y) coordinates in the map's resolution are within the bounds of the grid
        return (0 <= xy[:, 0]) & (xy[:, 0] < self.width_px) & (0 <= xy[:, 1]) & (xy[:, 1] < self.height_px)
    
    def xy_r_to_ij(self, xy: NDArray) -> NDArray:
        squeeze_back = len(xy.shape) == 1
        if squeeze_back:
            xy = xy[np.newaxis, :]
        
        # Floor to get discrete cell indices
        xy_floor = np.floor(xy).astype(np.int32)

        ij = np.zeros_like(xy_floor)
        ij[:, 0] = self.height_px - xy_floor[:, 1] - 1  # i = row (top = 0)
        ij[:, 1] = xy_floor[:, 0]                       # j = col

        if squeeze_back:
            return np.squeeze(ij)

        return ij
    
    def xy_m_to_xy_r(self, xy: NDArray) -> NDArray:
        # Convert (x, y) coordinates in meters to (x, y) coordinates in the map's resolution
        squeeze_back = len(xy.shape) == 1
        if squeeze_back:
            xy = xy[np.newaxis, :]
        
        xy_r = np.zeros_like(xy, dtype=np.float32)
        xy_r[:, 0] = xy[:, 0] / self.resolution
        xy_r[:, 1] = xy[:, 1] / self.resolution
        
        if squeeze_back:
            return np.squeeze(xy_r)
        
        return xy_r
    
    def xy_m_to_ij(self, xy: NDArray) -> NDArray:
        # Convert (x, y) coordinates in meters to row-column (i, j) indices into the map
        squeeze_back = len(xy.shape) == 1
        if squeeze_back:
            xy = xy[np.newaxis, :]
        
        xy_r_floored = np.floor(xy / self.resolution).astype(np.int32)
        
        ij = np.zeros_like(xy_r_floored)
        ij[:, 0] = self.height_px - xy_r_floored[:, 1] - 1
        ij[:, 1] = xy_r_floored[:, 0]
        
        if squeeze_back:
            return np.squeeze(ij)
        
        return ij
    
    def ij_to_xy_m(self, ij: NDArray) -> NDArray:
        # Convert row-column (i, j) indices into the map to (x, y) coordinates in meters
        squeeze_back = len(ij.shape) == 1
        if squeeze_back:
            ij = ij[np.newaxis, :]
        
        xy_m = np.zeros_like(ij, dtype=np.float32)
        xy_m[:, 0] = (ij[:, 1] + 0.5) * self.resolution
        xy_m[:, 1] = (self.height_px - ij[:, 0] - 1 + 0.5) * self.resolution
        
        if squeeze_back:
            return np.squeeze(xy_m)
        
        return xy_m
    
    def ij_to_xy_r(self, ij: NDArray) -> NDArray:
        # Convert row-column (i, j) indices into the map to (x, y) coordinates in the map's resolution
        squeeze_back = len(ij.shape) == 1
        if squeeze_back:
            ij = ij[np.newaxis, :]
        
        xy_r = np.zeros_like(ij, dtype=np.float32)
        xy_r[:, 0] = ij[:, 1] + 0.5
        xy_r[:, 1] = self.height_px - ij[:, 0] - 1 + 0.5
        
        if squeeze_back:
            return np.squeeze(xy_r)
        
        return xy_r
    

class OccupancyGridMapper:
    def __init__(self,
                 resolution: float,
                 width_m: float,
                 height_m: float,
                 p_hit: float = 0.9,
                 p_miss: float = 0.1,
                 max_height_px=None,
                 max_width_px=None) -> None:
        # Resolution refers to the size of each cell in the grid [m]        
        self.l_occ = np.log(p_hit / (1 - p_hit))  # Log-odds for occupied cell
        self.l_free = np.log(p_miss / (1 - p_miss))  # Log-odds for free cell
        self.l_unknown = np.log(0.5 / (1 - 0.5))  # Log-odds for unknown cell
        
        self.l_occ_max = np.log(0.99 / (1 - 0.99))  # Log-odds for max occupied cell
        self.l_free_min = np.log(0.01 / (1 - 0.01))  # Log-odds for max free cell
        
        self.resolution = resolution
        
        height_px, width_px = self.compute_map_size(resolution, height_m, width_m)
        
        if max_height_px and max_width_px:
            assert max_height_px >= height_px and max_width_px >= width_px, \
                f"Map size ({height_px, width_px}) with resolution {resolution} exceeds the maximum size: {max_height_px, max_width_px}"
            self.height_px = max_height_px
            self.width_px = max_width_px
        else:
            self.height_px = height_px
            self.width_px = width_px
        
        self.height_m = self.height_px * self.resolution
        self.width_m = self.width_px * self.resolution
        
        self.grid = np.full((self.height_px, self.width_px), self.l_unknown, dtype=np.float32)
        
        self._indexer = ArrayIndexer(resolution, self.height_px, self.width_px)
        
        # TODO: see if pre-computing the bresenham lines makes this faster
        # origin = 
        # self._bresenham_lines = bresenhamline()
    
    @staticmethod
    def compute_map_size(resolution: float, height_m: float, width_m: float) -> tuple[int, int]:
        """
        Compute the size of the occupancy grid map based on resolution and dimensions.
        
        Args:
            resolution (float): The size of each cell in the grid in meters.
            width_m (float): The width of the grid in meters.
            height_m (float): The height of the grid in meters.
        
        Returns:
            tuple[int, int]: The height and width of the grid in cells.
        """
        return int(np.ceil(height_m / resolution)), int(np.ceil(width_m / resolution))
    
    @property
    def free_area_m2(self, p_occupied_cutoff=0.2) -> int:
        log_odds_cutoff = np.log(p_occupied_cutoff / (1 - p_occupied_cutoff))
        return np.count_nonzero(self.grid < log_odds_cutoff) * self.resolution**2
    
    def reset(self) -> None:
        """
        Reset the occupancy grid to unknown state.
        """
        self.grid.fill(self.l_unknown)
    
    def to_prob_map(self) -> NDArray:
        """
        Convert log-odds to probability.
        
        Returns:
            NDArray: Probability values.
        """
        clipped_grid = np.clip(self.grid, self.l_free_min, self.l_occ_max)
        expl = np.exp(clipped_grid)
        return expl / (1.0 + expl)  # Convert log-odds to probability
    
    def process_scan(self, pose: NDArray, scan_points_b_B: NDArray) -> None:
        """
        Process a single scan and update the occupancy grid.
        
        Args:
            pose (NDArray): The robot's pose (x, y, theta).
            scan_points_b_B (NDArray): The (x, y) points from the laser scan in body frame.
        """
        # Rotate scan points into world frame
        W_R_B = spatialmath.base.rot2(pose[2])
        scan_points_w_W = scan_points_b_B @ W_R_B.T + pose[:2]  # shape (N, 2)
        
        # Convert start and end points of each ray to map-relative coordinates
        scan_points_xy_r = self._indexer.xy_m_to_xy_r(scan_points_w_W)
        pose_xy_r = self._indexer.xy_m_to_xy_r(pose[:2])
        
        scan_points_xy_r_floored = np.floor(scan_points_xy_r).astype(np.int32)
        pose_xy_r_floored = np.floor(pose_xy_r).astype(np.int32)

        # Now cast rays from robot position to each scan endpoint
        # First floor the start and end points to get the grid indices
        _start_points = pose_xy_r_floored[None, :] * np.ones_like(scan_points_xy_r_floored)
        out = bresenhamline(_start_points, scan_points_xy_r_floored, max_iter=-1)

        rays_xy_r_floored, max_iter = out
        rays_xy_r_floored = rays_xy_r_floored.reshape((-1, max_iter, 2))

        # Compute distances along the ray        
        ray_dists = np.linalg.norm(rays_xy_r_floored - pose_xy_r_floored, ord=1, axis=-1)
        scan_dists = np.linalg.norm(scan_points_xy_r_floored - pose_xy_r_floored, ord=1, axis=-1)[:, np.newaxis]

        # Identify pre-hit and hit cells
        mask_pre = (ray_dists < scan_dists).flatten()
        mask_hit = (ray_dists == scan_dists).flatten()

        rays_xy_r_floored = rays_xy_r_floored.reshape((-1, 2))
        cell_xy_r_free = rays_xy_r_floored[mask_pre]
        cell_xy_r_hit = rays_xy_r_floored[mask_hit]

        # Filter out-of-bounds
        mask_in_bounds_free = self._indexer.xy_r_in_bounds(cell_xy_r_free)
        mask_in_bounds_hit = self._indexer.xy_r_in_bounds(cell_xy_r_hit)
        cell_xy_r_free = cell_xy_r_free[mask_in_bounds_free]
        cell_xy_r_hit = cell_xy_r_hit[mask_in_bounds_hit]

        # Convert to integer grid indices
        grid_ij_free = self._indexer.xy_r_to_ij(cell_xy_r_free)
        grid_ij_hit = self._indexer.xy_r_to_ij(cell_xy_r_hit)

        fast_add_at(self.grid, grid_ij_free, self.l_free)
        fast_add_at(self.grid, grid_ij_hit, self.l_occ)

        # Also mark robot's cell as free
        pose_ij = self._indexer.xy_r_to_ij(pose_xy_r)
        self.grid[pose_ij[0], pose_ij[1]] += self.l_free


if __name__ == "__main__":
    import cProfile
    import pstats
    import io
    
    og_mapper = OccupancyGridMapper(resolution=0.5, width_m=4, height_m=4)
    
    # Test the indexing
    xy = np.array([[0.0, 0.0],
                   [1.0, 1.0],
                   [0.5, 3.5],
                   [2.4, 1.2]])
    ij = og_mapper._indexer.xy_m_to_ij(xy)
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
    
    og_mapper.process_scan(pose, points)
    og_mapper.reset()

    profiler = cProfile.Profile(timeunit=1)
    profiler.enable()
    og_mapper.process_scan(pose, points)
    profiler.disable()

    # Collect stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumtime')  # or 'tottime'
    stats.print_stats()

    # Print output
    print(stream.getvalue())
    
    