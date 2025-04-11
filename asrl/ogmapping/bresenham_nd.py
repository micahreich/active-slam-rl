"""
N-D Bresenham line algo
https://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
"""
import numpy as np


def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype), max_iter

def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    out, max_iter = _bresenhamlines(start, end, max_iter)
    return out.reshape(-1, start.shape[-1]), max_iter


if __name__ == "__main__":
    angles = np.deg2rad(np.arange(0, 360, 45))
    end_points = np.rint(np.column_stack([
        5.0 * np.cos(angles),
        5.0 * np.sin(angles)
    ])).astype(np.int32)
    
    start_points = np.zeros_like(end_points)
    
    print("Start points:")
    print(start_points)
    print("End points:")
    print(end_points)
    
    # start_points = np.array([
    #     [0, 0],
    #     [0, 0]
    # ])
    # end_points = np.array([
    #     [5, 5],
    #     [0, 1]
    # ])
    
    point_distances = np.ceil(np.linalg.norm(end_points - start_points, axis=-1)).astype(np.int32)
    
    bresenham_points, max_iter = bresenhamline(start_points, end_points, max_iter=-1)
    bresenham_points_distances = np.ceil(np.linalg.norm(bresenham_points - start_points[0], axis=-1)).astype(np.int32)
    
    scanline_intervals_indices = []
    scanline_intervals = []
    start_idx = 0

    for threshold in point_distances:
        # Find where arr[start_idx:] < threshold
        mask = bresenham_points_distances[start_idx:start_idx + max_iter] >= threshold
        
        if np.any(mask):
            # First match after start_idx
            rel_idx = np.argmax(mask)  # first True
            idx = start_idx + rel_idx
            scanline_interval = np.arange(start_idx, idx+1)
            scanline_intervals.append(range(scanline_interval[0], scanline_interval[-1]+1))
            scanline_intervals_indices.append(scanline_interval)
            start_idx += max_iter  # start next search after this index
        else:
            scanline_intervals.append(None)  # or -1 or np.nan to indicate not found
            break  # no further search possible
    
    scanline_intervals_indices = np.concatenate(scanline_intervals_indices)
    
    print(point_distances)
    print(bresenham_points_distances)    
    print(bresenham_points)
    print(scanline_intervals)
    
    for i, interval in enumerate(scanline_intervals):
        print(f"endpoint {i}:           {end_points[i]}")
        print(f"bresenham endpoint {i}: {bresenham_points[interval[-1]]}")
    
    # log_odds_update = np.ones(max_iter, dtype=np.float32) * self.l_free
    print(scanline_intervals_indices)
    print(bresenham_points[scanline_intervals_indices])
    
    
    # distances = np.linalg.norm(bresenham_points, axis=-1)
    
    # print(distances)
    
    # breakpoints = np.where(distances[:-1] > distances[1:])
    # print(breakpoints)
    