from typing import List
import numpy as np
from numpy.typing import NDArray
from spatialmath.base import *
from scipy.ndimage import binary_dilation, generate_binary_structure
from asrl.ogmapping import utils
from asrl.ogmapping.bresenham import trace_all_beams, apply_logodds_updates
from asrl.ogmapping.utils import ArrayIndexer, log_odds, transform_points
import numba


class OccupancyGridMapper:
    def __init__(self,
                 resolution: float,
                 width_m: float,
                 height_m: float,
                 p_hit: float = 0.9,
                 p_miss: float = 0.1,
                 max_height_px=None,
                 max_width_px=None) -> None:
        self.l_occ = log_odds(p_hit)
        self.l_free = log_odds(p_miss)
        self.l_unknown = log_odds(0.5)
        
        self.l_max = log_odds(1 - 1e-5)
        self.l_min = log_odds(1e-5)
        
        # Resolution refers to the size of each cell in the grid [m]
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
        
        self.indexer = ArrayIndexer(resolution, self.height_px, self.width_px)
    
    @staticmethod
    def compute_map_size(resolution: float, height_m: float, width_m: float) -> tuple[int, int]:
        return int(np.ceil(height_m / resolution)), int(np.ceil(width_m / resolution))
    
    @property
    def free_area_m2(self, p_occupied_cutoff=0.2) -> int:
        log_odds_cutoff = log_odds(p_occupied_cutoff)
        return np.count_nonzero(self.grid < log_odds_cutoff) * self.resolution**2
    
    @property
    def entropy(self) -> float:
        p = self.to_prob_map()
        entropies = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        total_entropy = np.sum(entropies)
        
        return total_entropy / (self.height_px * self.width_px)
    
    def reset(self) -> None:
        self.grid.fill(self.l_unknown)
    
    def to_prob_map(self) -> NDArray:
        clipped_grid = np.clip(self.grid, self.l_min, self.l_max)
        
        return 1 / (1.0 + np.exp(-clipped_grid))  # Convert log-odds to probability
    
    def process_scans(self, poses: NDArray, scans_B_BP_2d: NDArray, masks: NDArray, n_rays_per_scan: int) -> None:
        is_batched = poses.ndim == 2
        if not is_batched:
            poses = poses[None, ...]
            scans_B_BP_2d = scans_B_BP_2d[None, ...]
        
        B, _ = poses.shape
        
        assert scans_B_BP_2d.shape == (B, n_rays_per_scan, 2), \
            f"Scans shape {scans_B_BP_2d.shape} must be (B, n_rays_per_scan, 2)"
        assert scans_B_BP_2d.shape[0] == poses.shape[0], \
            f"Scans shape {scans_B_BP_2d.shape[0]} must match poses shape {poses.shape[0]}"
        
        W_T_B = np.stack([xyt2tr(p) for p in poses], axis=0)
        scans_W_WP_2d = transform_points(W_T_B, scans_B_BP_2d)

        # Convert from meters to pixel indices
        starts_xy_m = np.repeat(poses[:, :2], n_rays_per_scan, axis=0)
        ends_xy_m = np.reshape(scans_W_WP_2d, (B * n_rays_per_scan, 2))
        
        starts = self.indexer.xy_m_to_xy_r(starts_xy_m).astype(np.int32)
        ends = self.indexer.xy_m_to_xy_r(ends_xy_m).astype(np.int32)
        masks = np.reshape(masks, (B * n_rays_per_scan,))
        
        max_cells = np.amax(np.amax(np.abs(ends - starts), axis=1))
        
        cells, lengths = trace_all_beams(starts, ends, max_cells)
        apply_logodds_updates(self.grid, cells, lengths, masks, self.l_free, self.l_occ)
         
        # # Clip the log-odds values to the maximum and minimum thresholds for numerical stability
        # np.clip(self.grid, self.l_min, self.l_max, out=self.grid)
        
    def frontiers_mask(self, free_thresh=0.35, occ_thresh=0.65):
        l_free = log_odds(free_thresh)
        l_occ = log_odds(occ_thresh)
        
        is_free = self.grid < l_free
        is_obstacle = self.grid > l_occ
        is_unknown = ~is_free & ~is_obstacle
                
        unknown_dilated = binary_dilation(is_unknown, generate_binary_structure(rank=2, connectivity=1))
        obstacles_dilated = binary_dilation(is_obstacle, generate_binary_structure(rank=2, connectivity=2))
        
        frontiers = unknown_dilated & is_free & ~obstacles_dilated
        return frontiers
    
    def sample_frontiers(self, k, output_type='xy_m'):
        assert output_type in ['xy_m', 'ij'], \
            f"output_type must be 'xy_m' or 'ij', but got {output_type}"
        
        frontiers_map = self.frontiers_mask()
        frontiers_ij = np.argwhere(frontiers_map > 0)
        
        if len(frontiers_ij) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        fps_samples = utils.fast_fps(frontiers_ij, k)
        
        if output_type == 'xy_m':
            return self.indexer.ij_to_xy_m(fps_samples)
        else:
            return fps_samples
