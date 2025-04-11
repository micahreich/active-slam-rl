import numpy as np
from numpy.typing import NDArray
from spatialmath.base import angdiff
import matplotlib.pyplot as plt

from asrl.ogmapping.og_map import OccupancyGridMapper
from asrl.slam_sim.array_map import ArrayMap


class SimulationEnvironment:
    def __init__(self,
                 array_map: ArrayMap = None,
                 og_map_resolution: float = 0.1,
                 omega: float = 1.0,
                 v: float = 1.0,
                 dt=0.1,
                 travel_cut_short_dist_m: float = 0.1):
        self.pose = np.zeros(3)  # Initial pose
        self.omega = omega  # Angular velocity
        self.v = v  # Linear velocity
        self.dt = dt  # Time step
        self.travel_cut_short_dist_m = travel_cut_short_dist_m
        self.array_map = array_map
        self.og_map = OccupancyGridMapper(og_map_resolution,
                                          width_m=self.array_map.resolution * self.array_map.width,
                                          height_m=self.array_map.resolution * self.array_map.height)
    
    def reset(self) -> None:
        """
        Reset the simulation environment to a random pose in free space and reset the occupancy grid map.
        """
        pose = np.empty((3,))
        pose[:2] = self.array_map.sample_free_space(output_type='xy_m')
        pose[2] = np.random.uniform(0, 2 * np.pi)
        
        self.pose = pose
        self.og_map.reset()
    
    def step(self, action: NDArray) -> None:
        # Given an action, step the sim forward until action is terminated
        angle, distance = action
        max_travel_distance = self.array_map.max_travel_distance_along_ray(self.pose[:2], self.pose[2] + angle)
        travel_distance = max(0, min(max_travel_distance, distance) - self.travel_cut_short_dist_m)
        
        traveled_poses = self.travel_along_ray(angle, travel_distance)
        scans = self.array_map.raycast_in_map(traveled_poses)
        
        for scan in scans:
            self.og_map.update(scan)
        
        self.pose = traveled_poses[-1]
    
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
        
        # Determine poses while turning to the desired angle
        T = abs(angle) / abs(self.omega)
        N = int(np.ceil(T / self.dt))
        t = np.linspace(self.dt, N * self.dt, N)
        
        poses_turn = np.empty((N, 3))
        poses_turn[:, 2] = theta0 + np.sign(angle) * np.minimum(abs(self.omega) * t, abs(angle))
        poses_turn[:, :2] = self.pose[:2]
    
        # Determine poses while moving straight
        ray = np.array([
            np.cos(theta0 + angle),
            np.sin(theta0 + angle)
        ])
        
        T = distance / self.v
        N = int(np.ceil(T / self.dt))
        t = np.linspace(self.dt, N * self.dt, N)
        ds = np.minimum(self.v * t, distance)
        
        poses_straight = np.empty((N, 3))
        poses_straight[:, 2] = theta0 + angle
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
    array_map = ArrayMap('/home/dev/workspace/asrl/maps/floorplan1.txt', resolution=1, verbose=True)
    env = SimulationEnvironment(array_map, omega=1.0, v=1.0, dt=0.25)
    env.pose = np.array([2.0, 1.0, np.pi/2])  # Initial pose
    angle = np.pi / 4  # 45 degrees
    distance = 1.0
    poses = env.travel_along_ray(angle, distance)
    
    print(f"Traveled poses: {poses}")
    
    ax = plot_poses(poses)
    plt.show()


if __name__ == "__main__":
    test_travel_along_ray()