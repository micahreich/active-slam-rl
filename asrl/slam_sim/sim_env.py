import os
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from spatialmath.base import angdiff, angle_wrap, wrap_0_2pi
import matplotlib.pyplot as plt

from asrl.ogmapping.og_map import OccupancyGridMapper
from asrl.slam_sim.array_map import ArrayMap
from asrl.slam_sim import MAPS_DIRECTORY
from asrl.ogmapping.utils import log_odds
from skimage.graph import route_through_array


class SimulationEnvironment:
    def __init__(self,
                 map_name,
                 og_map_resolution,
                 dt,
                 og_map_shape=(None, None),
                 np_random=np.random):
        self._dt = dt  # Time step
        
        self._r_min_m = 0.0
        self._r_max_m = 6.0
        self._range_noise_m = 0.0

        map_name, _ = os.path.splitext(map_name)
        map_fpath = os.path.join(MAPS_DIRECTORY, f"{map_name}.txt")
        
        self.array_map = ArrayMap(map_fpath,
                                  np_random=np_random)
        
        self.og_map = OccupancyGridMapper(og_map_resolution,
                                          width_m=self.array_map.width_m,
                                          height_m=self.array_map.height_m,
                                          p_hit=0.8,
                                          p_miss=0.2,
                                          max_height_px=og_map_shape[-2],
                                          max_width_px=og_map_shape[-1])
    
    def reset(self, pose=None) -> Tuple[NDArray, NDArray]:
        self.timesteps_elapsed = 0
        
        if pose is None:
            coord_xy_m = self.array_map.sample_free_space(output_type='xy_m')
            self.pose = np.array([coord_xy_m[0], coord_xy_m[1], 0.0])
        else:
            self.pose = pose
            
        self.og_map.reset()
        
        # Perform a raycast to update the occupancy grid map just with the initial pose
        initial_scan, n_rays_per_scan = self.array_map.raycast_in_map(
            self.pose,
            r_min_m=self._r_min_m,
            r_max_m=self._r_max_m,
            range_noise_m=self._range_noise_m,
        )
        
        self.og_map.process_scans(self.pose, initial_scan, n_rays_per_scan)
    
    def step(self, action: NDArray) -> None:
        self.timesteps_elapsed += 1

        r_goal = int( action[0] * (self.og_map.height_px - 1) )
        c_goal = int( action[1] * (self.og_map.width_px - 1) )
        
        # Decide if agent can go here or not based on the occupancy grid map        
        if self.og_map.grid[r_goal, c_goal] >= log_odds(0.4):
            # Wants to move into occupied or unknown space
            return None
        
        r_curr, c_curr = self.og_map.indexer.xy_m_to_ij(self.pose[:2])

        # Move to the new position by planning a path
        prob_grid = self.og_map.to_prob_map()
                
        path_ij, cost = route_through_array(
            array=prob_grid,
            start=(r_curr, c_curr),
            end=(r_goal, c_goal),
            fully_connected=True,
            geometric=True
        )

        traversed_poses_xy_m = np.zeros((len(path_ij), 3))
        traversed_poses_xy_m[:, :2] = self.og_map.indexer.ij_to_xy_m(np.asarray(path_ij))

        # Update the occupancy grid map with scans along the path
        scans, n_rays_per_scan = self.array_map.raycast_in_map(
            traversed_poses_xy_m,
            r_min_m=self._r_min_m,
            r_max_m=self._r_max_m,
            range_noise_m=self._range_noise_m,
        )
        
        self.og_map.process_scans(traversed_poses_xy_m, scans, n_rays_per_scan)
        
        # Update the agent's pose
        self.pose = traversed_poses_xy_m[-1]
        
        return traversed_poses_xy_m

    def visualize_map_and_agent(self, fig, ax,
                                traversed_poses = None):
        ax.clear()
        
        grid_prob = self.og_map.to_prob_map()
        extent = [0, self.og_map.width_m, 0, self.og_map.height_m]
        
        grid_prob_img = ax.imshow(
            grid_prob,
            cmap='gray_r', interpolation='nearest',
            origin='upper', extent=extent
        )
        
        agent = ax.scatter([self.pose[0]], [self.pose[1]], c='red', s=100, marker='x')
        
        if traversed_poses is not None:
            path = ax.plot(
                traversed_poses[:, 0], traversed_poses[:, 1],
                color='blue', linewidth=2, label='Path'
            )

        xticks = np.arange(0, self.og_map.width_m, 1.0)
        yticks = np.arange(0, self.og_map.height_m, 1.0)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        plt.title(f'Occupancy grid map (H={self.og_map.entropy:.4f})')
        
        return grid_prob_img
        

def test1():
    env = SimulationEnvironment(
        map_name='box2',
        og_map_resolution=0.2,
        dt=0.1,
    )
    
    env.reset()
    
    fig, ax = plt.subplots()
    
    canvas = env.visualize_map_and_agent(fig, ax)
    fig.colorbar(canvas, ax=ax, label='Probability')
    
    def on_click(event):
        if event.inaxes:
            r_goal_n = 1.0 - event.ydata / env.og_map.height_m
            c_goal_n = event.xdata / env.og_map.width_m
            
            print(f"Clicked at: ({r_goal_n:.2f}, {c_goal_n:.2f})")
            
            traversed_poses = env.step(np.array([r_goal_n, c_goal_n]))
            env.visualize_map_and_agent(fig, ax, traversed_poses)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.show()
    
if __name__ == "__main__":
    test1()