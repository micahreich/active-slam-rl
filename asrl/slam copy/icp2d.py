import numpy as np
from typing import Optional, Tuple, cast
from spatialmath.base import Points2, SE2Array, points2tr2
from numpy.typing import NDArray
import open3d as o3d


def ICP2d(
    src: Points2,
    dst: Points2,
    T: Optional[SE2Array] = None,
    max_corr_dist: float = 1.0,
) -> NDArray:
    if T is None:
        init = np.eye(4)
    else:
        init = np.eye(4)
        init[:2, :2] = T[:2, :2]
        init[:2, 3] = T[:2, 3]
        
    _src = np.zeros((src.shape[0], 3,), dtype=float)
    _src[:, :2] = src
    
    _dst = np.zeros((dst.shape[0], 3,), dtype=float)
    _dst[:, :2] = dst
    
    _src_points = o3d.utility.Vector3dVector(_src)
    _dst_points = o3d.utility.Vector3dVector(_dst)
    
    src_pc = o3d.geometry.PointCloud(_src_points)
    dst_pc = o3d.geometry.PointCloud(_dst_points)

    result = o3d.pipelines.registration.registration_icp(
        src_pc, dst_pc,                # source, target
        max_correspondence_distance=max_corr_dist,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    
    dst_T_src = result.transformation  # 4x4 transformation matrix
    
    # Extract the rotation and translation from the transformation matrix
    R = dst_T_src[:2, :2]
    t = dst_T_src[:2, 3]
    
    return (R, t), result


# def ICP2d(
#     reference: Points2,
#     source: Points2,
#     T: Optional[SE2Array] = None,
#     max_iter: int = 20,
#     min_delta_err: float = 1e-4,
#     verbose: bool = False,
# ) -> SE2Array:
#     """
#     Iterated closest point (ICP) in 2D

#     :param reference: points (columns) to which the source points are to be aligned
#     :type reference: ndarray(2,N)
#     :param source: points (columns) to align to the reference set of points
#     :type source: ndarray(2,M)
#     :param T: initial pose , defaults to None
#     :type T: ndarray(3,3), optional
#     :param max_iter: max number of iterations, defaults to 20
#     :type max_iter: int, optional
#     :param min_delta_err: min_delta_err, defaults to 1e-4
#     :type min_delta_err: float, optional
#     :return: pose of source point cloud relative to the reference point cloud
#     :rtype: SE2Array

#     Uses the iterative closest point algorithm to find the transformation that
#     transforms the source point cloud to align with the reference point cloud, which
#     minimizes the sum of squared errors between nearest neighbors in the two point
#     clouds.

#     .. note:: Point correspondence is not required and the two point clouds do not have
#         to have the same number of points.

#     .. warning:: The point cloud argument order is reversed compared to :func:`points2tr`.

#     :seealso: :func:`points2tr`
#     """

#     # https://github.com/ClayFlannigan/icp/blob/master/icp.py
#     # https://github.com/1988kramer/intel_dataset/blob/master/scripts/Align2D.py
#     # hack below to use points2tr above
#     # use ClayFlannigan's improved data association

#     from scipy.spatial import KDTree

#     def _FindCorrespondences(
#         tree, source, reference
#     ) -> Tuple[NDArray, NDArray, NDArray]:
#         # get distances to nearest neighbors and indices of nearest neighbors
#         dist, indices = tree.query(source.T)

#         # remove multiple associatons from index list
#         # only retain closest associations
#         unique = False
#         matched_src = source.copy()
#         while not unique:
#             unique = True
#             for i, idxi in enumerate(indices):
#                 if idxi == -1:
#                     continue
#                 # could do this with np.nonzero
#                 for j in range(i + 1, len(indices)):
#                     if idxi == indices[j]:
#                         if dist[i] < dist[j]:
#                             indices[j] = -1
#                         else:
#                             indices[i] = -1
#                             break
#         # build array of nearest neighbor reference points
#         # and remove unmatched source points
#         point_list = []
#         src_idx = 0
#         for idx in indices:
#             if idx != -1:
#                 point_list.append(reference[:, idx])
#                 src_idx += 1
#             else:
#                 matched_src = np.delete(matched_src, src_idx, axis=1)

#         matched_ref = np.array(point_list).T

#         return matched_ref, matched_src, indices

#     mean_sq_error = 1.0e6  # initialize error as large number
#     delta_err = 1.0e6  # change in error (used in stopping condition)
#     num_iter = 0  # number of iterations
#     if T is None:
#         T = np.eye(3)

#     ref_kdtree = KDTree(reference.T)

#     source_hom = np.vstack((source, np.ones(source.shape[1])))

#     # tf_source = source
#     tf_source = cast(NDArray, T) @ source_hom
#     tf_source = tf_source[:2, :]

#     while delta_err > min_delta_err and num_iter < max_iter:
#         # find correspondences via nearest-neighbor search
#         matched_ref_pts, matched_source, indices = _FindCorrespondences(
#             ref_kdtree, tf_source, reference
#         )

#         # find alignment between source and corresponding reference points via SVD
#         # note: svd step doesn't use homogeneous points
#         new_T = points2tr2(matched_source, matched_ref_pts)

#         # update transformation between point sets
#         T = T @ new_T

#         # apply transformation to the source points
#         tf_source = cast(NDArray, T) @ source_hom
#         tf_source = tf_source[:2, :]

#         # find mean squared error between transformed source points and reference points
#         # TODO: do this with fancy indexing
#         new_err = 0
#         for i in range(len(indices)):
#             if indices[i] != -1:
#                 diff = tf_source[:, i] - reference[:, indices[i]]
#                 new_err += np.dot(diff, diff.T)

#         new_err /= float(len(matched_ref_pts))

#         # update error and calculate delta error
#         delta_err = abs(mean_sq_error - new_err)
#         mean_sq_error = new_err
#         if verbose: print("ITER", num_iter, delta_err, mean_sq_error)

#         num_iter += 1

#     return T