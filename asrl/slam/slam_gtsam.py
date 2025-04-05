import gtsam
import numpy as np
from numpy.typing import NDArray
from spatialmath import SE2

from asrl.slam.icp import icp

# Define noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(3 * np.array([1.0, 1.0, 1.0]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))

class GraphICPSLAM2DGTSAM:
    def __init__(self) -> None:
        self._scans = []
        self._poses = gtsam.Values()
        self._last_transform = np.eye(3)

        self._graph = gtsam.NonlinearFactorGraph()

    def step(self, scan: NDArray[np.floating]) -> None:
        pose_id = len(self._scans)

        if 0 == pose_id:
            self._scans.append(scan)
            # this is the first pose in the graph so we can't add an edge

            pose = gtsam.Pose2(np.zeros(3))
            self._graph.add(gtsam.PriorFactorPose2(pose_id, pose, PRIOR_NOISE))
            self._poses.insert(pose_id, pose)

            return

        previous_pose_id = pose_id - 1
        previous_scan = self._scans[previous_pose_id]
        transform, _, _ = icp(previous_scan, scan, self._last_transform)

        if np.linalg.norm(transform.t) < 0.1:
            self._last_transform = transform
            return

        self._last_transform = np.eye(3)
        self._scans.append(scan)

        previous_pose = self._poses.atPose2(previous_pose_id)
        pose = gtsam.Pose2((transform @ SE2(previous_pose.matrix())).xyt())
        self._poses.insert(pose_id, pose)

        factor = gtsam.BetweenFactorPose2(pose_id, previous_pose_id, gtsam.Pose2(transform.xyt()), ODOMETRY_NOISE)
        self._graph.add(factor)

    def poses(self) -> NDArray[np.floating]:
        optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._poses)
        self._poses = optimizer.optimize()
        vertices = len(self._scans)
        poses = np.zeros((vertices, 3))
        for pose_id in range(vertices):
            pose = self._poses.atPose2(pose_id)
            poses[pose_id, :2] = pose.translation()
            poses[pose_id, 2] = pose.theta()
        return poses

    def hessian(self) -> NDArray[np.floating]:
        linear = self._graph.linearize(self._poses)
        H, i = linear.hessian()
        return H
