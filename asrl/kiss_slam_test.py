import kiss_slam
from kiss_slam.pipeline import SlamPipeline
from kiss_icp.pipeline import OdometryPipeline
import pickle
import numpy as np


class Dataset:
    def __init__(self, scans_pkl: str) -> None:
        self.sequence_id = 0
        with open(scans_pkl, 'rb') as f:
            self.lidar_scans = pickle.load(f)

    def __len__(self):
        return len(self.lidar_scans)
    
    def __getitem__(self, idx):
        scan_np = self.lidar_scans[idx]
        n_heights = 10
        dh = 0.1
        
        scans = [np.copy(scan_np) for _ in range(n_heights)]
        for i in range(1, n_heights):
            scans[i][:, 2] += i * dh
        
        scan_np = np.concatenate(scans, axis=0)
        
        n_points = scan_np.shape[0]
        scan_timestamps = idx * 0.1 * np.ones(n_points)
        return scan_np.astype(np.float64), scan_timestamps.astype(np.float64)


if __name__ == '__main__':    
    dataset = Dataset('/home/dev/workspace/asrl/open3d_tests/lidar_scans.pkl')
    i0, _ = dataset[0]

    pipeline = OdometryPipeline(dataset)
    results = pipeline.run()
    
    print(results)