import numpy as np

def parse_carmen_corrected_log(filename, laser_name="FLASER"):
    poses = []       # list of (x, y, theta)
    scans = []       # list of range arrays
    angles = np.deg2rad(np.arange(-90, 90, 1, dtype=np.float32))

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(laser_name):
                tokens = line.strip().split()
                start_idx = 1
                
                n_readings = int(tokens[start_idx])
                start_idx += 1
                
                # Laser range readings
                ranges = np.array(list(map(float, tokens[start_idx:start_idx + n_readings])))
                start_idx += n_readings
                
                # Laser pose (robot pose where laser was fired)
                pose_2d = np.array(list(map(float, tokens[start_idx:start_idx + 3])))
                start_idx += 3
                
                poses.append(pose_2d)
                scans.append(ranges)
                
    return np.array(poses), np.array(scans), angles

if __name__ == "__main__":
    filename = "asrl/ogmapping/intel.log"
    poses, scans, angles = parse_carmen_corrected_log(filename)
    
    print(f"Poses shape: {poses.shape}")
    print(f"Scans shape: {scans.shape}")
    print(f"Angles shape: {angles.shape}")
    
    # Turn scans into point clouds
    cos_theta, sin_theta = np.cos(angles), np.sin(angles)
    xs = scans * cos_theta
    ys = scans * sin_theta
    points = np.stack((xs, ys), axis=-1)

    