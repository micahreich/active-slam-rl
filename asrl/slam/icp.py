# Original code: https://github.com/ClayFlannigan/icp
# Modified to reject pairs that have greater distance than the specified threshold

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from spatialmath import SE2


def best_fit_transform(A: NDArray[np.floating], B: NDArray[np.floating]) -> NDArray[np.floating]:
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    '''

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def nearest_neighbor(src: NDArray[np.floating], dst: NDArray[np.floating]) -> tuple[
    NDArray[np.floating], NDArray[np.floating]]:
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(
        A: NDArray[np.floating],
        B: NDArray[np.floating],
        init_pose: Optional[NDArray[np.floating]] = None,
        max_iter: int = 20,
        max_dist: float = np.inf,
        tolerance: float = 0.001,
) -> tuple[SE2, NDArray[np.floating], int]:
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert max_iter > 0
    assert A.shape[1] == B.shape[1]

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((A.shape[0], m + 1))
    dst = np.ones((B.shape[0], m + 1))
    src[:, :m] = A
    dst[:, :m] = B

    # apply the initial pose estimation
    if init_pose is not None:
        T = SE2(init_pose).A
    else:
        T = SE2().A

    prev_error = np.inf

    for i in range(max_iter):
        src_current = src @ T.T

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src_current[:, :m], dst[:, :m])

        # Reject pairs that have max_dist between them
        matches_filtered = distances < max_dist
        src_filtered = src_current[matches_filtered, :m]
        dst_filtered = dst[indices[matches_filtered], :m]

        # compute the transformation between
        # the current source and nearest destination points
        T_new = best_fit_transform(src_filtered, dst_filtered)

        T = T_new @ T

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # try:
    T = SE2(T, check=False)
    # except:
    #     print(T)
    #     raise RuntimeError("failed to converge")

    return T, distances, i + 1
