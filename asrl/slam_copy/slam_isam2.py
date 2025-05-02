import gtsam
import numpy as np
from numpy.typing import NDArray
from asrl.slam_copy.icp2d import ICP2d
from scipy.spatial import KDTree


# Define noise models
xy_sigma = 0.1
theta_sigma = np.deg2rad(5)
sigma = np.array([xy_sigma, xy_sigma, theta_sigma])

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(0.5 * sigma)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(sigma)
LOOP_NOISE = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber(1.0),
    gtsam.noiseModel.Diagonal.Sigmas(0.1 * sigma)
)

class GraphSLAMiSAM2:
    def __init__(self,
                 initial_pose: NDArray = np.zeros(3),
                 min_translation_delta_m: float = 0.3,
                 min_rotation_delta_rad: float = np.deg2rad(5),
                 max_loop_closure_rad_m: float = 2.0,
                 min_loop_closure_steps: int = 5,):
        self.initial_pose = gtsam.Pose2(initial_pose)
        self.min_translation_delta_m = min_translation_delta_m
        self.min_rotation_delta_rad = min_rotation_delta_rad
        self.max_loop_closure_rad_m = max_loop_closure_rad_m
        self.min_loop_closure_steps = min_loop_closure_steps
        
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        
        self._isam = gtsam.ISAM2(parameters)
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_estimates = gtsam.Values()
        
        self._scans = []
        self._last_transform = None
        self._nn = None
        
    def step(self, scan: NDArray[np.floating]) -> None:
        pose_id = len(self._scans)

        if pose_id == 0:
            self._scans.append(scan)

            self._graph.add(gtsam.PriorFactorPose2(pose_id, self.initial_pose, PRIOR_NOISE))
            self._poses.insert(pose_id, self.initial_pose)

            # this is the first pose in the graph so we can't add an edge
            return

        prev_id = pose_id - 1
        previous_scan = self._scans[prev_id]
        
        # ICP2d returns dst_T_src
        xyt, _, _ = ICP2d(
            src=scan,
            dst=previous_scan,
            T=self._last_transform,
            max_corr_dist=1.0,
        )
                
        # Check if the translation and rotation are significant
        prev_T_curr = gtsam.Pose2(xyt)
        translation = np.linalg.norm(prev_T_curr.translation())
        rotation = np.abs(prev_T_curr.theta())
        
        if translation < self.min_translation_delta_m and rotation < self.min_rotation_delta_rad:
            self._last_transform = prev_T_curr
            return
        else:
            self._last_transform = None
            
        current_estimate = self._isam.calculateEstimate()
        world_T_prev = current_estimate.atPose2(prev_id)
        world_T_curr = world_T_prev.compose(prev_T_curr)
        
        self._new_factors.add(
            gtsam.BetweenFactorPose2(
                prev_id, pose_id, prev_T_curr, ODOMETRY_NOISE
            )
        )
        
        self._new_estimates.insert(
            pose_id,
            world_T_curr
        )
        
        # Search for loop closures
        current_estimate_poses = self._values_to_poses(current_estimate)
        dists = np.linalg.norm(world_T_curr.translation() - current_estimate_poses[:, :2], axis=-1)
        dists_mask = dists < self.max_loop_closure_rad_m
        
        # Update iSAM estimate
        self._isam.update(self._new_factors, self._new_estimates)
        
        self._new_factors = gtsam.NonlinearFactorGraph()
        self._new_estimates = gtsam.Values()
    
    def _values_to_poses(self, values):
        n = values.size()
        poses = np.zeros((n, 3))
        
        for pose_id in range(n):
            pose = values.atPose2(pose_id)
            poses[pose_id, :2] = pose.translation()
            poses[pose_id, 2] = pose.theta()
            
        return poses
    
    def _build_nn(self, values=None):
        if values is None:
            values = self._isam.calculateEstimate()
        
        n_values = values.size()
        positions = np.zeros((n_values, 2), dtype=np.float32)
        
        for i in range(n_values):
            pose = values.atPose2(i)
            positions[i, :] = pose.translation()
        
        self._nn = KDTree(positions)