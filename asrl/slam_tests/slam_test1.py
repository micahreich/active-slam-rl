
import numpy as np
from asrl.slam.icp import icp
from asrl.slam.slam_gtsam import GraphICPSLAM2DGTSAM
from asrl.slam_sim.array_map import ArrayMap

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load("/home/dev/workspace/asrl/slam_tests/clicked_points.npz")
    clicked_points = data["points"]
    N_points, _ = clicked_points.shape
    
    clicked_poses = np.zeros((N_points, 3), dtype=np.float32)
    clicked_poses[:, :2] = clicked_points
    
    env = ArrayMap('/home/dev/workspace/asrl/maps/box2.txt', resolution=1.0)
    
    slam = GraphICPSLAM2DGTSAM(initial_pose=clicked_poses[0],
                               min_loop_closure_steps=N_points + 1)
    scans, mask, n_rays_per_scan = env.raycast_in_map(clicked_poses, r_min_m=0.0, r_max_m=np.inf,
                                                      range_noise_m=0.05)

    for i in range(len(scans)):
        scan = scans[i][mask[i]]
        
        assert not np.any(np.isnan(scan)) and not np.any(np.isinf(scan))
        
        # if i > 0:
        #     T, _, _ = icp(scans[i - 1], scans[i][:, :2])
        #     print(f"T ({i - 1} -> {i}): {T}")
        
        slam.step(scan)
    
    unoptimized_poses = slam.poses()
    
    slam.optimize()
    optimized_poses = slam.poses()
    
    # Plot the results
    plt.plot(unoptimized_poses[:, 0], unoptimized_poses[:, 1], 'r-', label='Unoptimized')
    plt.plot(optimized_poses[:, 0], optimized_poses[:, 1], 'b-', label='Optimized')
    plt.plot(clicked_poses[:, 0], clicked_poses[:, 1], 'g-', label='Clicked')
    plt.legend()
    plt.title('SLAM Optimization Results')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.show()