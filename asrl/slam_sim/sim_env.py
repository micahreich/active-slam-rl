import os
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from spatialmath.base import angdiff, angle_wrap, wrap_0_2pi
import matplotlib.pyplot as plt

from asrl.ogmapping.og_map import OccupancyGridMapper
from asrl.slam_sim.array_map import ArrayMap
from asrl.slam_sim import MAPS_DIRECTORY


class SimulationEnvironment:
    def __init__(self,
                 map_name,
                 og_map_resolution,
                 omega,
                 v,
                 dt,
                 travel_cut_short_dist_m,
                 og_map_shape):
        self.pose = np.zeros(3)  # Initial pose
        self._omega = omega  # Angular velocity
        self._v = v  # Linear velocity
        self._dt = dt  # Time step
        self._travel_cut_short_dist_m = travel_cut_short_dist_m
        self.timesteps_elapsed = 0
        self.envsteps_elapsed = 0
        
        self.r_max_m = 8.0
        
        map_name, _ = os.path.splitext(map_name)
        map_fpath = os.path.join(MAPS_DIRECTORY, f"{map_name}.txt")
        
        self.array_map = ArrayMap(map_fpath)
        self.og_map = OccupancyGridMapper(og_map_resolution,
                                          width_m=self.array_map.width_m,
                                          height_m=self.array_map.width_m,
                                          max_height_px=og_map_shape[-2],
                                          max_width_px=og_map_shape[-1])
    
    @property
    def time_elapsed(self) -> float:
        """
        Get the total time elapsed in the simulation.
        """
        return self.timesteps_elapsed * self._dt
    
    def reset(self, pose=None) -> Tuple[NDArray, NDArray]:
        """
        Reset the simulation environment to a random pose in free space and reset the occupancy grid map.
        """
        self.timesteps_elapsed = 0
        self.envsteps_elapsed = 0
        
        # if pose is None:
        #     pose = np.zeros((3,))
        #     pose[:2] = self.array_map.sample_free_space(output_type='xy_m')
        #     # pose[2] = np.random.uniform(0, 2 * np.pi)
        #     pose[2] = 0.0
        pose = np.array([2.0, 2.0, 0.0])
        
        self.pose = pose
        self.og_map.reset()
        
        # Perform a raycast to update the occupancy grid map just with the initial pose
        initial_scan = self.array_map.raycast_in_map(self.pose,
                                                     r_min_m=self.og_map.resolution,
                                                     r_max_m=self.r_max_m)
        self.og_map.process_scan(self.pose, initial_scan)
        
        return self.get_observation()
    
    # def _process_og_map(self, og_map: NDArray) -> NDArray:
    #     p_free_thresh = 0.1
    #     p_occ_thresh = 1 - p_free_thresh
        
    #     l_free_thresh = np.log(p_free_thresh / (1-p_free_thresh))
    #     l_occ_thresh = np.log(p_occ_thresh / (1-p_occ_thresh))
        
    #     occupied_cells = og_map > l_occ_thresh
    #     free_cells = og_map < l_free_thresh
        
    #     processed_map = np.where(occupied_cells, 0, -1)
    #     processed_map[free_cells] = 1
        
    #     return processed_map.astype(np.int8)
    
    def _process_og_map(self, og_map: OccupancyGridMapper) -> NDArray:
        processed_map = og_map.to_prob_map()
        processed_map = np.clip(processed_map * 255, 0, 255).astype(np.uint8)

        return processed_map
        
    def get_observation(self) -> Tuple[NDArray, NDArray]:
        """
        Get the current observation of the environment.
        
        Returns:
            Tuple[NDArray, NDArray]: The current pose and the occupancy grid map, both normalized to [0, 1];
            based on the map size for the pose and the map is converted to probabilities from log-odds.
        """
        normalized_pose = np.zeros(3)
        normalized_pose[0] = self.pose[0] / self.og_map.width_m
        normalized_pose[1] = self.pose[1] / self.og_map.height_m
        normalized_pose[2] = wrap_0_2pi(self.pose[2]) / (2 * np.pi)
        
        # prob_map = self.og_map.to_prob_map()
        # prob_map = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
        # prob_map_h, prob_map_w = self.og_map.shape
        processed_map = self._process_og_map(self.og_map)
        
        return processed_map[None, ...], normalized_pose
    
    def step(self, action: NDArray) -> None:
        """
        Advances the agent's state by executing the given action.

        The action consists of a relative angle (in radians) and a distance.
        The angle is interpreted in the agent's **body frame** — i.e., relative to the agent's current heading.
        The agent attempts to move along this direction for the specified distance, unless obstructed by the map.

        Parameters
        ----------
        action : NDArray
            A 2-element array `[angle, distance]`

        Notes
        -----
        - The agent's actual travel distance is capped by obstacles in the environment (via `max_travel_distance_along_ray`)
        and may be reduced by `travel_cut_short_dist_m`.
        - At each intermediate pose along the movement path, a raycast is performed and used to update the occupancy grid map.
        - The agent's final pose after movement is set to the last pose along the traveled ray.
        """
        angle, distance = action
        x0, y0, theta0 = self.pose
        angle = angle_wrap(angle, mode='-pi:pi')
        
        max_travel_distance = self.array_map.max_travel_distance_along_ray(self.pose[:2], self.pose[2] + angle)
        travel_distance = max(0, min(max_travel_distance - self._travel_cut_short_dist_m, distance))
        distance_from_env = max_travel_distance - distance
        
        # # Find occupancy at the ending position
        # ray = np.array([
        #     np.cos(theta0 + angle),
        #     np.sin(theta0 + angle)
        # ])
        
        # desired_end_position = self.pose[:2] + ray * distance
        # ogmap_i, ogmap_j = self.og_map._indexer.xy_m_to_ij(desired_end_position[0], desired_end_position[1])
        # occupancy_log_odds_end_position = self.og_map.grid[ogmap_i, ogmap_j]
        
        # Determine poses while moving straight
        traveled_poses = self.travel_along_ray(angle, travel_distance)
        timesteps_elapsed = len(traveled_poses)

        if timesteps_elapsed > 0:
            scans = self.array_map.raycast_in_map(traveled_poses,
                                                  r_min_m=self.og_map.resolution,
                                                  r_max_m=self.r_max_m)
            timesteps_elapsed = len(traveled_poses)
            
            for pose, scan in zip(traveled_poses, scans):
                self.og_map.process_scan(pose, scan)
            
            self.pose = traveled_poses[-1]
        else:
            timesteps_elapsed = 1
            
        self.timesteps_elapsed += timesteps_elapsed
        self.envsteps_elapsed += 1
        
        return self.get_observation(), timesteps_elapsed, distance_from_env
    
    def travel_along_ray(self, angle: NDArray, distance: float) -> NDArray:
        """
        Travel along a ray for a given distance.
        
        Args:
            angle (NDArray): The angle to turn before traveling.
            distance (float): The distance to travel along the ray.
        
        Returns:
            NDArray: The new position after traveling along the ray.
        """
        x0, y0, theta0 = self.pose
        
        distance_eps_m = 1e-5
    
        # Determine poses while moving straight
        if abs(distance) < distance_eps_m:
            return np.empty((0, 3))
        
        ray = np.array([
            np.cos(theta0 + angle),
            np.sin(theta0 + angle)
        ])
        
        T = distance / self._v
        N = int(np.ceil(T / self._dt))
        t = np.linspace(self._dt, N * self._dt, N)
        ds = np.minimum(self._v * t, distance)
        
        poses_straight = np.zeros((N, 3))
        poses_straight[:, :2] = self.pose[:2] + ray * ds[:, None]
        poses_straight[:, 2] = theta0
        
        return poses_straight

def plot_poses(poses, scale=0.2, ax=None):
    """
    Plot 2D poses (x, y, theta) in the plane.

    Args:
        poses: (B, 3) array of [x, y, theta] poses
        style: 'arrow', 'circle', or 'frame'
        scale: length of heading indicator
        ax: matplotlib axis (optional)
    """    
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

    for xi, yi, ti in poses:
        circle = plt.Circle((xi, yi), radius=scale * 0.5, edgecolor='black', facecolor='none')
        ax.add_patch(circle)
        ax.plot([xi, xi + scale * 0.5 * np.cos(ti)],
                [yi, yi + scale * 0.5 * np.sin(ti)], color='black')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    
    return ax


def test_travel_along_ray():
    env = SimulationEnvironment('/home/dev/workspace/asrl/maps/floorplan1.txt',
                                og_map_resolution=0.1,
                                omega=1.0,
                                v=1.0,
                                dt=0.1,
                                travel_cut_short_dist_m=0.1,
                                og_map_shape=(256, 256))
    env.pose = np.array([2.0, 1.0, 0.0])  # Initial pose
    angle = -np.pi / 4  # 45 degrees
    distance = 1.0
    poses = env.travel_along_ray(angle, distance)
    
    print(f"Traveled poses: {poses}")


if __name__ == "__main__":
    test_travel_along_ray()