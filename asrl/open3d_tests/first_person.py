import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

class FirstPersonExplorer:
    def __init__(self):
        self.window = gui.Application.instance.create_window("Open3D First Person", 1280, 720)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        self.speed = 0.05
        self.camera_pos = np.array([0.0, 1.0, 3.0])
        self.camera_forward = np.array([0.0, 0.0, -1.0])

        # Add a floor and a cube
        self._add_floor()
        self._add_cube([0, 0.5, 0])
        self._add_cube([2, 0.5, 2])
        self._add_cube([-2, 0.5, -2])

        self._update_camera()
        
        # Handle keyboard input
        self.window.set_on_key(self._on_key_event)

    def _add_floor(self):
        floor = o3d.geometry.TriangleMesh.create_box(width=10, height=0.1, depth=10)
        floor.translate([-5, -0.1, -5])
        floor.paint_uniform_color([0.8, 0.8, 0.8])
        self.scene.scene.add_geometry("floor", floor, rendering.MaterialRecord())

    def _add_cube(self, center):
        cube = o3d.geometry.TriangleMesh.create_box()
        cube.translate(center)
        cube.compute_vertex_normals()
        
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"  # <- Enables lighting and shading
        cube.paint_uniform_color([1, 0, 0])
        
        self.scene.scene.add_geometry(f"cube_{center}", cube, mat)

    def _update_camera(self):
        lookat = self.camera_pos + self.camera_forward
        self.scene.scene.camera.look_at(lookat, self.camera_pos, [0, 1, 0])

    def _rotate_yaw(self, degrees):
        # Rotate camera_forward vector around y-axis (up) by given degrees
        radians = np.deg2rad(degrees)
        c, s = np.cos(radians), np.sin(radians)
        x, z = self.camera_forward[0], self.camera_forward[2]
        self.camera_forward[0] = c * x + s * z
        self.camera_forward[2] = -s * x + c * z
        self.camera_forward = self.camera_forward / np.linalg.norm(self.camera_forward)

    def _on_key_event(self, event):
        if event.type == gui.KeyEvent.DOWN:
            direction = np.zeros(3)
            right = np.cross(self.camera_forward, [0, 1, 0])

            if event.key == gui.KeyName.W:
                direction += self.camera_forward
            elif event.key == gui.KeyName.S:
                direction -= self.camera_forward
            elif event.key == gui.KeyName.A:
                direction -= right
            elif event.key == gui.KeyName.D:
                direction += right
                
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                self.camera_pos += direction * self.speed
                self._update_camera()
                
            if event.key == gui.KeyName.LEFT:
                self._rotate_yaw(+2)  # degrees
                self._update_camera()
            elif event.key == gui.KeyName.RIGHT:
                self._rotate_yaw(-2)
                self._update_camera()
            

        return gui.Widget.EventCallbackResult.HANDLED.value


# Run the app
gui.Application.instance.initialize()
FirstPersonExplorer()
gui.Application.instance.run()
