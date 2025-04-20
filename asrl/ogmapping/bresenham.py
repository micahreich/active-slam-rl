import numpy as np
import numba
from numba import njit, prange
import matplotlib.pyplot as plt


@njit
def bresenham_2d(start, end):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    max_len = dx + dy + 1
    path = np.empty((max_len, 2), dtype=np.int32)
    i = 0

    while True:
        path[i, 0] = x
        path[i, 1] = y
        i += 1
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return path[:i]

@njit(parallel=True)
def trace_all_beams(starts, ends, max_cells):
    n_beams = starts.shape[0]
    cell_inds = -np.ones((n_beams, max_cells, 2), dtype=np.int32)
    lengths = np.zeros(n_beams, dtype=np.int32)

    for i in prange(n_beams):
        path = bresenham_2d(starts[i], ends[i])
        plen = path.shape[0]
        lengths[i] = plen
        for j in range(plen):
            cell_inds[i, j, 0] = path[j, 0]
            cell_inds[i, j, 1] = path[j, 1]
    return cell_inds, lengths

@njit
def apply_logodds_updates(map_grid, cell_inds, lengths, masks, log_free, log_occ):
    height, width = map_grid.shape
    
    n_beams = lengths.shape[0]
    for i in range(n_beams):
        L = lengths[i]
        
        for j in range(L - 1):
            x, y = cell_inds[i, j]
            
            if 0 <= x < width and 0 <= y < height:
                map_grid[height - y - 1, x] += log_free
                
        x, y = cell_inds[i, L - 1]
        end_hit = masks[i]
        
        if end_hit and 0 <= x < width and 0 <= y < height:
            map_grid[height - y - 1, x] += log_occ
        elif not end_hit and 0 <= x < width and 0 <= y < height:
            map_grid[height - y - 1, x] += log_free


if __name__ == "__main__":
    import time

    grid_shape = (256, 256)
    start = np.array([128, 128], dtype=np.int32)
    
    angles = np.deg2rad(np.arange(0, 360, 2))
    n_beams = len(angles)
    # 1.0 + 2.0 * np.cumsum(np.ones_like(angles)) / len(angles)
    distances = 8 + 64 * np.cumsum(np.ones_like(angles)) / len(angles)
        
    ends = np.rint(np.column_stack([
        start[0] + distances * np.cos(angles),
        start[1] + distances * np.sin(angles)
    ])).astype(np.int32)

    starts = np.tile(start, (n_beams, 1))
    
    # # Duplicate starts and ends
    # starts = np.tile(starts, (100, 1))
    # ends = np.tile(ends, (100, 1))

    # Max path length for prealloc
    max_cells = np.amax(np.amax(np.abs(ends - starts), axis=1))
    log_free = np.log(0.1 / (1 - 0.1))
    log_occ = np.log(0.9 / (1 - 0.9))

    # Warm-up
    map_grid = np.zeros(grid_shape)
    cells, lengths = trace_all_beams(starts, ends, max_cells)
    apply_logodds_updates(map_grid, cells, lengths, log_free=log_free, log_occ=log_occ)
    
    # Benchmark
    niter = 1000
    start_t = time.perf_counter()
    for _ in range(niter):
        map_grid = np.zeros(grid_shape)
        apply_logodds_updates(map_grid, cells, lengths, log_free=log_free, log_occ=log_occ)
    end_t = time.perf_counter()

    print(f"Update time (apply only): {1e3 * (end_t - start_t) / niter:.3f} ms")

    # Including beam tracing
    start_t = time.perf_counter()
    for _ in range(niter):
        map_grid = np.zeros(grid_shape)
        cells, lengths = trace_all_beams(starts, ends, max_cells)
        apply_logodds_updates(map_grid, cells, lengths, log_free=log_free, log_occ=log_occ)
    end_t = time.perf_counter()

    print(f"Total time per iteration (trace + update): {1e3 * (end_t - start_t) / niter:.3f} ms")
    
    expl = np.exp(map_grid)
    map_grid_probabilities = expl / (1.0 + expl)
    
    plt.imshow(map_grid_probabilities, cmap='gray_r')
    plt.show()
