import os
import matplotlib.pyplot as plt
import numpy as np

from asrl.slam_sim.array_map import ArrayMap


dragging = False
fig, ax = plt.subplots(figsize=(8, 8))
clicked_points = []

def on_press(event):
    global dragging
    if event.inaxes:
        dragging = True

def on_release(event):
    global dragging
    dragging = False

def on_motion(event):    
    if dragging and event.inaxes:
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        ax.plot(x, y, 'rx', markersize=4)
        fig.canvas.draw_idle()


if __name__ == "__main__":
    m = ArrayMap('/home/dev/workspace/asrl/maps/box2.txt', resolution=1.0)
    save_path="/home/dev/workspace/asrl/slam_tests/clicked_points.npz"

    ax.imshow(m.map_image, cmap='gray_r',
            interpolation='nearest',
            origin='upper',
            extent=[0, m.width_m, 0, m.height_m],
            vmin=0, vmax=1)
    
    ax.set_title("Click to select points on the map")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    plt.show()

    clicked_points_np = np.array(clicked_points, dtype=np.float32)
    np.savez_compressed(save_path, points=clicked_points_np)
    print(f"Saved {len(clicked_points)} point(s) to {os.path.abspath(save_path)}")

    print("All clicked points:", clicked_points)
    