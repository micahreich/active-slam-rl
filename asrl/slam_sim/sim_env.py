import os
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from spatialmath.base import angdiff, angle_wrap
import matplotlib.pyplot as plt

from asrl.ogmapping.og_map import OccupancyGridMapper
from asrl.slam_sim.array_map import ArrayMap
from asrl.slam_sim import MAPS_DIRECTORY


class SimulationEnvironment:
    def __init__(self,
                 map_name: str = "floorplan1",
                 og_map_resolution: float = 0.1,
                 omega: float = 1.0,
                 v: float = 1.0,
                 dt=0.1,
                 travel_cut_short_dist_m: float = 0.1,
                 map_image_size_px: Tuple[int, int] = (64, 64)):
        self.pose = np.zeros(3)  # Initial pose
        self._omega = omega  # Angular velocity
        self._v = v  # Linear velocity
        self._dt = dt  # Time step
        self._travel_cut_short_dist_m = travel_cut_short_dist_m
        self.timesteps_elapsed = 0
        
        map_name, _ = os.path.splitext(map_name)
        map_fpath = os.path.join(MAPS_DIRECTORY, f"{map_name}.txt")
        
        self.array_map = ArrayMap(map_fpath)        
        self.og_map = OccupancyGridMapper(og_map_resolution,
                                          width_m=self.array_map.width_m,
                                          height_m=self.array_map.width_m,
                                          max_height_px=map_image_size_px[0],
                                          max_width_px=map_image_size_px[1])
    
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
        
        if pose is None:
            pose = np.zeros((3,))
            pose[:2] = self.array_map.sample_free_space(output_type='xy_m')
            pose[2] = 0.0 #np.random.uniform(0, 2 * np.pi)
        
        self.pose = pose
        self.og_map.reset()
        
        # Perform a raycast to update the occupancy grid map just with the initial pose
        initial_scan = self.array_map.raycast_in_map(self.pose)
        self.og_map.process_scan(self.pose, initial_scan)
        
        return self.get_observation()
    
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
        normalized_pose[2] = angle_wrap(self.pose[2], mode='0:2pi') / 2*np.pi
        
        return normalized_pose, self.og_map.to_prob_map()
    
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
        angle = angle_wrap(angle, mode='-pi:pi')
        
        max_travel_distance = self.array_map.max_travel_distance_along_ray(self.pose[:2], self.pose[2] + angle)
        travel_distance = max(0, min(max_travel_distance - self._travel_cut_short_dist_m, distance))
        
        traveled_poses = self.travel_along_ray(angle, travel_distance)
        timesteps_elapsed = len(traveled_poses)

        if timesteps_elapsed > 0:
            scans = self.array_map.raycast_in_map(traveled_poses)
            timesteps_elapsed = len(traveled_poses)
            
            for pose, scan in zip(traveled_poses, scans):
                self.og_map.process_scan(pose, scan)
            
            self.pose = traveled_poses[-1]
        else:
            timesteps_elapsed = 1
            
        self.timesteps_elapsed += timesteps_elapsed
        return self.get_observation(), timesteps_elapsed
    
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
        
        angle_eps_rad = 1e-5
        distance_eps_m = 1e-5
        
        # Determine poses while turning to the desired angle
        if abs(angle) < angle_eps_rad:
            poses_turn = np.empty((0, 3))
        else:
            T = abs(angle) / abs(self._omega)
            N = int(np.ceil(T / self._dt))
            t = np.linspace(self._dt, N * self._dt, N)
            angles = np.sign(angle) * np.minimum(abs(self._omega) * t, abs(angle))
            
            poses_turn = np.empty((N, 3))
            poses_turn[:, 2] = angle_wrap(theta0 + angles, mode='0:2pi')
            poses_turn[:, :2] = self.pose[:2]
    
        # Determine poses while moving straight
        if abs(distance) < distance_eps_m:
            poses_straight = np.empty((0, 3))
        else:
            ray = np.array([
                np.cos(theta0 + angle),
                np.sin(theta0 + angle)
            ])
            
            T = distance / self._v
            N = int(np.ceil(T / self._dt))
            t = np.linspace(self._dt, N * self._dt, N)
            ds = np.minimum(self._v * t, distance)
            
            poses_straight = np.empty((N, 3))
            poses_straight[:, 2] = angle_wrap(theta0 + angle, mode='0:2pi')
            poses_straight[:, :2] = self.pose[:2] + ray * ds[:, None]
            
        # Combine the two segments
        poses = np.vstack((poses_turn, poses_straight))
        
        return poses

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
    env = SimulationEnvironment('/home/dev/workspace/asrl/maps/floorplan1.txt', omega=1.0, v=1.0, dt=0.25,
                                map_image_size_px=(256, 256))
    env.pose = np.array([2.0, 1.0, np.pi/2])  # Initial pose
    angle = -np.pi / 4  # 45 degrees
    distance = 1.0
    poses = env.travel_along_ray(angle, distance)
    
    print(f"Traveled poses: {poses}")
    
    ax = plot_poses(poses)
    plt.show()


if __name__ == "__main__":
    test_travel_along_ray()