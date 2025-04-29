# Original code: https://github.com/ClayFlannigan/icp
# Modified to reject pairs that have greater distance than the specified threshold

from typing import Optional

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


def apply(pose1: NDArray[np.float64], pose2: NDArray[np.float64]) -> NDArray[np.float64]:
    c = np.cos(pose1[2])
    s = np.sin(pose1[2])
    return np.array([
        pose1[0] + c * pose2[0] - s * pose2[1],
        pose1[1] + s * pose2[0] + c * pose2[1],
        pose1[2] + pose2[2],
    ])


def inverse(pose: NDArray[np.float64]) -> NDArray[np.float64]:
    theta = -pose[2]
    c = -np.cos(theta)
    s = -np.sin(theta)
    return np.array([
        c * pose[0] - s * pose[1],
        s * pose[0] + c * pose[1],
        theta,
    ])


def best_fit_transform(A: NDArray[np.floating], B: NDArray[np.floating]) -> NDArray[np.floating]:
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: pose delta [x, y, t]
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

    return np.r_[t, np.arctan2(R[1, 0], R[0, 0])]


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

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(
        A: NDArray[np.floating],
        B: NDArray[np.floating],
        pose: Optional[NDArray[np.floating]] = None,
        max_iter: int = 200,
        max_dist: float = np.inf,
        tolerance: float = 0.001,
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
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
    if pose is None:
        pose = np.zeros(3)

    prev_error = np.inf

    for i in range(max_iter):
        src_current = src @ gtsam.Pose2(pose).matrix().T

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src_current[:, :m], dst[:, :m])

        # Reject pairs that have max_dist between them
        matches_filtered = distances < max_dist
        src_filtered = src_current[matches_filtered, :m]
        dst_filtered = dst[indices[matches_filtered], :m]

            # plt.gca().set_aspect('equal')
            # plt.scatter(src_current[:, 0], src_current[:, 1], c='r', alpha=0.1)
            # plt.scatter(dst[:, 0], dst[:, 1], c='b', alpha=0.1)
            # plt.scatter(src_filtered[:, 0], src_filtered[:, 1], c='r')
            # plt.scatter(dst_filtered[:, 0], dst_filtered[:, 1], c='b')
            # plt.plot(
            #     np.c_[src_filtered[:, 0], dst_filtered[:, 0]].T,
            #     np.c_[src_filtered[:, 1], dst_filtered[:, 1]].T,
            #     '-o'
            # )
            # plt.show()

        # compute the transformation between
        # the current source and nearest destination points
        shift = best_fit_transform(src_filtered, dst_filtered)

        pose = apply(shift, pose)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return pose, distances, i + 1
