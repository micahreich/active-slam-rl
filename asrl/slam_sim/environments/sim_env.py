import numpy as np
import open3d as o3d
from typing import Tuple, Dict
from asrl.slam.slam_gtsam import GraphICPSLAM2DGTSAM
from asrl.slam.icp import icp
from asrl.open3d_tests.env_builder import ArrayMap, collect_body_lidar_scans
from spatialmath import SE2
from asrl.ogmapping.ogmapper import *

class Agent:
    def __init__(self, pose: np.ndarray):
        self.pose = pose.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        dx, dy = action
        theta = self.pose[2]
        delta = np.array([
            dx,
            dy,
            0.0
        ])
        self.pose += delta
        return self.pose.copy()
    
    def predict_move(self, action: np.ndarray) -> np.ndarray:
        dx, dy = action
        theta = self.pose[2]
        delta = np.array([
            dx,
            dy,
            0.0
        ])
        predicted_pose = self.pose.copy()+ delta
        return predicted_pose

    def reset(self, pose: np.ndarray):
        self.pose = pose.copy()


class SimEnv:
    def __init__(self, occupancy_grid: np.ndarray, start_pose: np.ndarray, scan_resolution: int = 360, update_occupancy: bool = False):
        self.map = ArrayMap(occupancy_grid, scale=1)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.agent = Agent(start_pose)
        self.scan_resolution = scan_resolution

        for cube in self.map.to_o3d_geometry():
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(cube)
            self.scene.add_triangles(mesh)

        self.slam = GraphICPSLAM2DGTSAM(self.agent.pose, min_loop_closure_steps=5)
        self.last_scan = self._raycast_scan()
        self.slam.step(self.last_scan[:, :2])
        self.mapper = OccupancyGridMapper(resolution=0.2, width_m=12.0, height_m=12.0)
        self.update_occupancy = update_occupancy

    def _raycast_scan(self) -> np.ndarray:
        scan = collect_body_lidar_scans(np.array([self.agent.pose]), self.scene)[0]
        # Remove any rows with NaNs or Infs
        mask = ~np.isnan(scan).any(axis=1) & ~np.isinf(scan).any(axis=1)
        scan = scan[mask]

        return scan

    def reset(self, pose: np.ndarray):
        self.agent.reset(pose)
        self.slam = GraphICPSLAM2DGTSAM(self.agent.pose)
        self.last_scan = self._raycast_scan()
        self.slam.step(self.last_scan[:, :2])
        self.mapper.reset_grid()

    def step(self, action: np.ndarray) -> Tuple[Dict, float]:
        new_pose = self.agent.predict_move(action)
        new_coord = self.map.xy_to_coord(new_pose[:2], self.map.free_space.shape)

        if(self.map.free_space[tuple(new_coord)]!=0):
            self.agent.step(action)
        else:
            new_pose = self.agent.pose

        scan = self._raycast_scan()
        self.slam.step(scan[:, :2])

        # Update occupancy grid 
        if self.update_occupancy:
            self.update_occupancy_grid()

        obs = {
            "pose_graph": self.slam._graph,
            "current_scan": scan,
            "agent_pose": new_pose,  
            "occupancy_grid": self.mapper.log_odds_map_to_prob_map(),
        }
        H = self.slam.hessian()
        reward = np.log(np.linalg.det(H + 1e-6 * np.eye(H.shape[0])))

        return obs, reward
    
    def update_occupancy_grid(self):
        self.mapper.reset_grid()
        est_poses = self.slam.poses()
        for pose, scan in zip(est_poses, self.slam._scans):
            self.mapper.process_scan(pose, scan)

