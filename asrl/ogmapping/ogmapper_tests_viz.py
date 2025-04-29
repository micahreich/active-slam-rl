import numpy as np
from asrl.ogmapping import utils
from asrl.ogmapping.og_map import OccupancyGridMapper
import matplotlib.pyplot as plt
from time import perf_counter
from spatialmath.base import *
import scipy
from asrl.ogmapping.utils import transform_points
from asrl.slam_sim.array_map import ArrayMap


if __name__ == "__main__":
    env = ArrayMap('/home/dev/workspace/asrl/maps/floorplan1.txt', resolution=1.0, verbose=True)
    height, width = env._walls.shape
    
    og_map = OccupancyGridMapper(
        0.2,
        width,
        height,
        p_hit=0.8, p_miss=0.2
    )
    
    # Plot with imshow and interactive backend
    fig, ax = plt.subplots()
    
    extent = [0, og_map.width_m, 0, og_map.height_m]
    
    im = ax.imshow(og_map.to_prob_map(), cmap='gray_r', origin='upper', vmin=0.0, vmax=1.0, extent=extent)
    frontiers = ax.imshow(np.ma.zeros_like(og_map.grid), cmap='Reds', alpha=0.5, origin='upper', vmin=0.0, vmax=1.0, extent=extent)
    sampled_frontiers = ax.scatter([], [], c='blue', marker='x', s=10, label='Sampled Frontiers')
    
    # Callback function to handle mouse click
    def on_click(event):
        if event.inaxes != ax:
            return
        
        print(f"Clicked at: ({event.xdata :.2f}, {event.ydata :.2f})")
        
        pose = np.array([event.xdata, event.ydata, 0.0])
        
        scans_B_BP_2d, t_hit_mask, n_rays = env.raycast_in_map(pose, r_max_m=4.0)
        og_map.process_scans(pose, scans_B_BP_2d, t_hit_mask, n_rays)
        
        frontiers_map = og_map.frontiers_mask()
        masked_frontiers_map = np.ma.masked_where(frontiers_map == 0, frontiers_map)

        samples = og_map.sample_frontiers(k=10, output_type='xy_m')

        # Update display
        im.set_data(og_map.to_prob_map())
        frontiers.set_data(masked_frontiers_map)
        sampled_frontiers.set_offsets(samples)
        
        fig.canvas.draw_idle()  # Efficient redraw

    # Connect event handler
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()