import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from spatialmath import SE2

from asrl.open3d_tests.env_builder import *
from asrl.slam.icp import icp
from asrl.slam.slam import *


og = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
])

m = ArrayMap(og, scale=1)

m.to_o3d_geometry()
scene = o3d.t.geometry.RaycastingScene()
walls = m.to_o3d_geometry()
for cube in walls:
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cube))

poses = np.array([
    [2.5, 1.5, 0],
    [2.5, 1.8, 0.8],
    [2.5, 2.0, 0.0],
    [2.5, 2.2, 0.0],
])

scans = collect_body_lidar_scans(poses, scene)

A = scans[0][:, :2]
B = scans[1][:, :2]
C = scans[2][:, :2]
D = scans[3][:, :2]

T, _, _ = icp(A, B, max_dist=1.0)

A_new = (np.c_[A, np.ones(A.shape[0])] @ SE2(T).A.T)[:, :2]

plt.gca().set_aspect('equal')
# plt.scatter(A[:,0], A[:,1], c='r')
plt.scatter(B[:,0], B[:,1], c='b')
plt.scatter(A_new[:,0], A_new[:,1], c='g')

slam = GraphICPSLAM2D()
slam.step(A)
slam.step(B)
slam.step(C)
slam.step(D)
slam.poses()
print(slam.hessian())
