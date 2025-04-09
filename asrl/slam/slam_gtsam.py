import gtsam
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KDTree

from asrl.slam.icp import icp

# Define noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(1e-3 * np.array([1.0, 1.0, 1.0]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))


class GraphICPSLAM2DGTSAM:
    def __init__(
            self,
            initial_pose: NDArray = np.zeros(3),
            min_pose_delta: float = 0.5,
            max_loop_closure_distance: float = 2.5,
            min_loop_closure_steps: int = 10,
    ) -> None:
        self.initial_pose = gtsam.Pose2(*initial_pose)
        self.min_pose_delta = min_pose_delta
        self.max_loop_closure_distance = max_loop_closure_distance
        self.min_loop_closure_steps = min_loop_closure_steps

        self._scans = []
        self._poses = gtsam.Values()
        self._last_transform = np.zeros(3)
        self._steps_since_loop_closure = 0
        self._steps_since_nn = 0

        self._graph = gtsam.NonlinearFactorGraph()
        self._nn = None

    def step(self, scan: NDArray[np.floating]) -> None:
        pose_id = len(self._scans)

        if 0 == pose_id:
            self._scans.append(scan)

            self._graph.add(gtsam.PriorFactorPose2(pose_id, self.initial_pose, PRIOR_NOISE))
            self._poses.insert(pose_id, self.initial_pose)

            # this is the first pose in the graph so we can't add an edge
            return

        previous_pose_id = pose_id - 1
        previous_scan = self._scans[previous_pose_id]

        previous_scan = previous_scan[~np.isnan(previous_scan).any(axis=1)]
        previous_scan = previous_scan[~np.isinf(previous_scan).any(axis=1)]

        scan = scan[~np.isnan(scan).any(axis=1)]
        scan = scan[~np.isinf(scan).any(axis=1)]

        # Only run ICP if both scans are non-empty
        if previous_scan.shape[0] == 0 or scan.shape[0] == 0:
            return

        icp_result = icp(previous_scan, scan, self._last_transform, max_dist=0.5)
        if icp_result is not None:
            transform, _, _ = icp_result
        else:
            return

        if np.linalg.norm(transform[:2]) < self.min_pose_delta:
            self._last_transform = transform
            return

        self._last_transform = np.zeros(3)
        self._scans.append(scan)

        previous_pose = self._poses.atPose2(previous_pose_id)
        pose = previous_pose * gtsam.Pose2(transform).inverse()
        self._poses.insert(pose_id, pose)

        factor = gtsam.BetweenFactorPose2(pose_id, previous_pose_id, gtsam.Pose2(transform), ODOMETRY_NOISE)
        self._graph.add(factor)

        self._steps_since_loop_closure += 1
        self._steps_since_nn += 1
        if self._steps_since_nn > self.min_loop_closure_steps:
            self._update_nn()
        if self._steps_since_loop_closure > self.min_loop_closure_steps:
            self._check_loop_closure(pose_id)


    def _update_nn(self):
        poses = self.poses()[:-self.min_loop_closure_steps, :2]
        poses = poses[~np.isnan(poses).any(axis=1)]  # Remove rows with NaNs
        self._nn = KDTree(poses)
        self._steps_since_nn = 0

    def _check_loop_closure(self, pose_id) -> None:
        if self._nn is None:
            return
        pose = self._poses.atPose2(pose_id)
        query_pt = pose.translation()
        if np.isnan(query_pt).any():
            return
        [indices], _ = self._nn.query_radius([pose.translation()], self.max_loop_closure_distance, return_distance=True,
                                             sort_results=True)
        if indices.shape[0] == 0:
            return
        for closure_id in indices:
            scan = self._scans[pose_id]
            closure_pose = self._poses.atPose2(closure_id)
            closure_scan = self._scans[closure_id]

            closure_scan = closure_scan[~np.isnan(closure_scan).any(axis=1)]
            closure_scan = closure_scan[~np.isinf(closure_scan).any(axis=1)]
            scan = scan[~np.isnan(scan).any(axis=1)]
            scan = scan[~np.isinf(scan).any(axis=1)]
            if closure_scan.shape[0] == 0 or scan.shape[0] == 0:
                return
            transform = pose.between(closure_pose)
            if np.isnan(transform.translation()).any() or np.isnan(transform.theta()):
                return
            
            icp_result = icp(closure_scan, scan, np.r_[transform.translation(), transform.theta()], max_dist=1.0)
            if icp_result is not None:
                transform, distances, _ = icp_result
            else:
                return

            #print(f"Checking LC {pose_id} <-> {closure_id}, mean ICP error: {np.mean(distances):.4f}")
            if np.mean(distances) > 0.1:
                continue
            print(f"[LOOP CLOSURE] Between poses {pose_id} and {closure_id} (ICP mean error: {np.mean(distances):.4f})")
            factor = gtsam.BetweenFactorPose2(pose_id, closure_id, gtsam.Pose2(transform), ODOMETRY_NOISE)
            self._graph.add(factor)
            self.optimize()
            self._steps_since_loop_closure = 0
            return

    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._poses)
        self._poses = optimizer.optimize()

    def poses(self) -> NDArray[np.floating]:
        n = len(self._scans)
        poses = np.zeros((n, 3))
        for pose_id in range(n):
            pose = self._poses.atPose2(pose_id)
            poses[pose_id, :2] = pose.translation()
            poses[pose_id, 2] = pose.theta()
        return poses

    def hessian(self) -> NDArray[np.floating]:
        linear = self._graph.linearize(self._poses)
        H, i = linear.hessian()
        return H
