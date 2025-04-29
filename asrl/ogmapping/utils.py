from typing import List
import numpy as np
from numpy.typing import NDArray
import spatialmath
import numba    
    

def transform_points(T, points):
    R = T[:, :2, :2]
    t = T[:, :2, 2]
    rotated = points @ np.transpose(R, (0, 2, 1))
    transformed = rotated + t[:, None, :]
    
    return transformed

def log_odds(p):
    return np.log( p / (1 - p) )

def fast_fps(points, k):
    N = len(points)
    
    if k >= N:
        return points
    
    selected = [np.random.randint(N)]
    dists = np.full(N, np.inf)

    for _ in range(1, k):
        last = points[selected[-1]]
        dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))
        selected.append(np.argmax(dists))

    return points[selected]

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

    # def ij_in_bounds(self, ij: NDArray) -> NDArray:
    #     # Check if (i, j) indices are within the bounds of the grid
    #     return (0 <= ij[:, 0]) & (ij[:, 0] < self.height_px) & (0 <= ij[:, 1]) & (ij[:, 1] < self.width_px)
    
    # def xy_r_in_bounds(self, xy: NDArray) -> NDArray:
    #     # Check if (x, y) coordinates in the map's resolution are within the bounds of the grid
    #     return (0 <= xy[:, 0]) & (xy[:, 0] < self.width_px) & (0 <= xy[:, 1]) & (xy[:, 1] < self.height_px)
    
    # def xy_r_to_ij(self, xy: NDArray) -> NDArray:
    #     squeeze_back = len(xy.shape) == 1
    #     if squeeze_back:
    #         xy = xy[np.newaxis, :]
        
    #     # Floor to get discrete cell indices
    #     xy_floor = np.floor(xy).astype(np.int32)

    #     ij = np.zeros_like(xy_floor)
    #     ij[:, 0] = self.height_px - xy_floor[:, 1] - 1  # i = row (top = 0)
    #     ij[:, 1] = xy_floor[:, 0]                       # j = col

    #     if squeeze_back:
    #         return np.squeeze(ij)

    #     return ij
    
    def xy_m_to_xy_r(self, xy: NDArray) -> NDArray:
        # Convert (x, y) coordinates in meters to (x, y) coordinates in the map's resolution
        squeeze_back = len(xy.shape) == 1
        if squeeze_back:
            xy = xy[np.newaxis, :]
        
        xy_r = np.zeros_like(xy, dtype=np.float32)
        xy_r[:, 0] = xy[:, 0] / self.resolution
        xy_r[:, 1] = xy[:, 1] / self.resolution
        
        if squeeze_back:
            return np.squeeze(xy_r, axis=0)
        
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
            return np.squeeze(ij, axis=0)
        
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
            return np.squeeze(xy_m, axis=0)
        
        return xy_m
    
    # def ij_to_xy_r(self, ij: NDArray) -> NDArray:
    #     # Convert row-column (i, j) indices into the map to (x, y) coordinates in the map's resolution
    #     squeeze_back = len(ij.shape) == 1
    #     if squeeze_back:
    #         ij = ij[np.newaxis, :]
        
    #     xy_r = np.zeros_like(ij, dtype=np.float32)
    #     xy_r[:, 0] = ij[:, 1] + 0.5
    #     xy_r[:, 1] = self.height_px - ij[:, 0] - 1 + 0.5
        
    #     if squeeze_back:
    #         return np.squeeze(xy_r)
        
    #     return xy_r

if __name__ == "__main__":
    poses = np.random.randn(10, 3)
    poses = np.array([spatialmath.base.xyt2tr(p) for p in poses])
    
    points = np.random.randn(10, 12, 2)
    
    points_transformed = transform_points(poses, points)
    
    R0 = poses[0, :2, :2]
    t0 = poses[0, :2, 2]
    points0 = points[0]
    
    print(
        (R0 @ points0.T + t0[:, None]).T
    )
    
    print(points_transformed[0])
    
    