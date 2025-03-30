import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load the .ply file as a legacy TriangleMesh
mesh_legacy = o3d.io.read_triangle_mesh('/home/dev/workspace/asrl/open3d_tests/stata_03.ply')
mesh_legacy.compute_vertex_normals()  # Optional but often useful

print("Number of vertices:", len(mesh_legacy.vertices))
print("Number of triangles:", len(mesh_legacy.triangles))

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[45, 35, 8])

o3d.visualization.draw_geometries([mesh_legacy, axis])

# # Convert the legacy mesh to a tensor-based mesh for raycasting
# mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

# # Visualize the mesh in an interactive window
# o3d.visualization.draw_geometries([mesh_tensor])


# # Create a raycasting scene and add the mesh
# scene = o3d.t.geometry.RaycastingScene()
# scene.add_triangles(mesh_tensor)

# # Generate rays using a pinhole camera model
# rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
#     fov_deg=90,
#     center=[0, 0, 0],
#     eye=[0, 0, -5],
#     up=[0, 1, 0],
#     width_px=640,
#     height_px=480,
# )

# # Perform raycasting
# ans = scene.cast_rays(rays)

# # Visualize the t_hit values (depth map)
# plt.imshow(ans['t_hit'].numpy(), cmap='jet')
# plt.title("Raycasting Depth Map")
# plt.colorbar(label="Distance")
# plt.show()