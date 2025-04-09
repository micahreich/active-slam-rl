import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from asrl.slam_sim.environments.sim_env import SimEnv

known_keymaps = [
    'keymap.fullscreen',
    'keymap.grid',
    'keymap.home',
    'keymap.back',
    'keymap.forward',
    'keymap.pan',
    'keymap.zoom',
    'keymap.save',
    'keymap.quit',
    'keymap.xscale',
    'keymap.yscale',
    'keymap.toggle',
    'keymap.yscale', 
    'keymap.xscale',
]

# Disable only those that exist in this version
for keymap in known_keymaps:
    if keymap in matplotlib.rcParams:
        matplotlib.rcParams[keymap] = []

# Define occupancy grid (1 = obstacle, 0 = free)
og = np.array([
    [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1],
    [1,0,1,1,0,1,1,1,0,1, 1,0,1,1,1,0,1,1,0,1],
    [1,0,1,1,0,1,1,1,0,1, 0,0,1,1,0,0,1,1,0,1],
    [1,0,1,0,0,1,0,1,1,1, 1,1,0,0,1,1,1,0,1,1],
    [1,1,1,1,1,1,0,1,0,1, 1,1,1,1,0,1,0,1,0,1],
    [1,0,1,1,0,0,1,1,0,1, 0,0,1,1,0,1,1,1,0,1],
    [1,0,1,1,1,1,0,0,0,1, 1,1,1,0,0,1,1,0,0,1],
    [1,0,1,1,1,1,0,1,1,1, 1,0,1,1,1,1,0,1,1,1],
    [1,1,1,1,0,0,1,1,0,1, 1,0,0,1,1,0,1,1,0,1],
    [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,0,0,1,1,0,1, 1,0,0,1,1,0,1,1,1,1],
    [1,0,1,1,1,1,0,1,1,1, 1,0,1,1,1,1,0,1,1,1],
    [1,0,1,1,1,1,0,0,0,1, 1,1,1,0,0,1,1,0,0,1],
    [1,0,1,1,0,0,1,1,0,1, 0,0,1,1,0,1,1,1,0,1],
    [1,1,1,1,1,1,0,1,0,1, 1,1,1,1,0,1,0,1,0,1],
    [1,0,1,1,0,1,1,1,0,1, 1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,1,1,1,0,0,1, 1,1,0,0,1,1,0,0,0,1],
    [1,1,1,1,1,1,0,1,1,1, 0,1,1,1,1,1,0,1,1,1],
    [1,0,1,0,0,1,1,1,0,1, 0,1,1,1,0,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1],
])

# Initialize environment

start_pose = np.array([5.5, 5.5, 0])
env = SimEnv(og, start_pose, update_occupancy=True)

# Define discrete action set (dx, dy in body frame)
actions = {
    'w': np.array([0.0, 0.4]),     # forward (along Y)
    's': np.array([0.0, -0.4]),    # backward
    'a': np.array([-0.4, 0.0]),    # left
    'd': np.array([0.4, 0.0]),     # right
    'q': np.array([-0.3, 0.3]),    # forward-left
    'e': np.array([0.3, 0.3]),     # forward-right
    'z': np.array([-0.3, -0.3]),   # backward-left
    'c': np.array([0.3, -0.3]),    # backward-right
}

# Set up real-time visualization
plt.ion()
fig, ax = plt.subplots()

free_space = env.map.free_space
H, W = free_space.shape
crop_min_i, _, crop_min_j, _ = env.map.env_bounds
offset = np.array([crop_min_j, crop_min_i])

ax.imshow(free_space, origin='upper', cmap='gray', extent=(0, W, 0, H))

scan_plot = ax.scatter([], [], c='r', s=5)
pose_plot = ax.scatter([], [], c='b', s=20)

# Lock the axes so they never autoscale
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.set_autoscale_on(False)
fig.tight_layout()

plt.title("Use WASDQEZC keys to move. Close window to exit.")

map_fig, map_ax = plt.subplots()
map_img = map_ax.imshow(env.mapper.log_odds_map_to_prob_map(), cmap='gray', origin='upper')
map_ax.set_title("Occupancy Grid Map (SLAM Estimated)")
map_ax.set_aspect('equal')
plt.tight_layout()

def on_key(event):
    key = event.key
    if key not in actions:
        return

    # Step environment
    action = actions[key]
    obs, reward = env.step(action)
    #print(f"gtsam pose: {env.slam.poses()[-1]}")
    scan = obs['current_scan']
    pose = obs['agent_pose']

    scan_pts = scan[:, :2]
    # Remove any NaNs or Infs
    scan_pts = scan_pts[~np.isnan(scan_pts).any(axis=1)]
    scan_pts = scan_pts[~np.isinf(scan_pts).any(axis=1)]

    # Plot updates
    scan_plot.set_offsets(scan_pts)
    pose_plot.set_offsets(pose[:2] + np.array([-0.5, 0.5]))

    prob_map = env.mapper.log_odds_map_to_prob_map()
    map_img.set_data(prob_map)
    map_img.set_clim(vmin=0.0, vmax=1.0)

    map_fig.canvas.draw_idle()
    map_fig.canvas.flush_events()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

cid = fig.canvas.mpl_connect('key_press_event', on_key)

plt.show(block=True)
