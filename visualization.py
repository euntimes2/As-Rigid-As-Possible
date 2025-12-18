import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering


def pick_handles(verts, faces):
    """
    Pick handle vertex indices using Open3D's GUI with drag selection.

    Controls:
        - Shift + Left Drag: box-select vertices
        - Shift + Left Click: pick a vertex (small box)
        - C: clear selection
        - Q or Esc: finish selection
    """
    print(f"Total vertices: {verts.shape[0]}")
    print("Shift+LMB drag to box-select, C to clear, Q/Esc to finish.")

    class HandlePicker:
        def __init__(self, verts, faces):
            self.verts = np.asarray(verts, dtype=np.float64)
            self.faces = np.asarray(faces, dtype=np.int32)
            self.selected = set()
            self._dragging = False
            self._drag_start = (0, 0)

            app = gui.Application.instance
            self.window = app.create_window("Pick handles (Shift+Drag)", 1024, 768)
            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            self.window.add_child(self.scene)

            self._init_scene()

            self.scene.set_on_mouse(self._on_mouse)
            self.window.set_on_key(self._on_key)

        def _init_scene(self):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(self.verts)
            mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            mesh.compute_vertex_normals()

            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.base_color = (0.7, 0.7, 0.75, 1.0)
            self.scene.scene.add_geometry("mesh", mesh, mat)

            bounds = mesh.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60.0, bounds, bounds.get_center())

        def _on_key(self, event):
            if event.type != gui.KeyEvent.Type.DOWN:
                return False

            if event.key in (gui.KeyName.Q, gui.KeyName.ESCAPE):
                self.window.close()
                return True

            if event.key == gui.KeyName.C:
                self.selected.clear()
                self._update_selection_geometry()
                print("Selection cleared.")
                return True

            return False

        def _on_mouse(self, event):
            if not event.is_modifier_down(gui.KeyModifier.SHIFT):
                return False

            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
                self._dragging = True
                self._drag_start = (event.x, event.y)
                return True

            if event.type == gui.MouseEvent.Type.BUTTON_UP and self._dragging:
                self._dragging = False
                drag_end = (event.x, event.y)
                self._select_in_rect(self._drag_start, drag_end)
                return True

            return False

        def _get_camera_matrices(self):
            cam = self.scene.scene.camera
            view = np.asarray(cam.get_view_matrix())
            try:
                proj = np.asarray(cam.get_projection_matrix())
            except TypeError:
                frame = self.scene.frame
                proj = np.asarray(cam.get_projection_matrix(frame.width, frame.height))
            return view, proj

        def _select_in_rect(self, start, end):
            x0, y0 = start
            x1, y1 = end
            min_x, max_x = sorted((x0, x1))
            min_y, max_y = sorted((y0, y1))

            if max_x - min_x < 4:
                min_x -= 2
                max_x += 2
            if max_y - min_y < 4:
                min_y -= 2
                max_y += 2

            frame = self.scene.frame
            width = max(1, frame.width)
            height = max(1, frame.height)

            view, proj = self._get_camera_matrices()
            verts_h = np.hstack([self.verts, np.ones((self.verts.shape[0], 1), dtype=np.float64)])
            clip = (proj @ view @ verts_h.T).T
            w = clip[:, 3:4]
            valid = w[:, 0] > 0
            ndc = clip[:, :3] / w

            in_view = (
                (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
                (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0) &
                (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
            )

            x = (ndc[:, 0] * 0.5 + 0.5) * width
            y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height

            in_rect = (
                (x >= min_x) & (x <= max_x) &
                (y >= min_y) & (y <= max_y)
            )

            picked = np.where(valid & in_view & in_rect)[0]
            if picked.size == 0:
                print("No vertices in the selection box.")
                return

            self.selected.update(picked.tolist())
            print(f"Selected {len(picked)} vertices. Total selected: {len(self.selected)}")
            self._update_selection_geometry()

        def _update_selection_geometry(self):
            try:
                self.scene.scene.remove_geometry("selection")
            except Exception:
                pass

            if not self.selected:
                return

            sel_pts = self.verts[np.array(sorted(self.selected), dtype=int)]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sel_pts)

            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 6.0
            mat.base_color = (1.0, 0.1, 0.1, 1.0)
            self.scene.scene.add_geometry("selection", pcd, mat)

    app = gui.Application.instance
    try:
        app.initialize()
    except Exception:
        pass

    picker = HandlePicker(verts, faces)
    app.run()

    if not picker.selected:
        print("No handle vertices were picked.")
        return np.array([], dtype=int)

    picked = sorted(picker.selected)
    print(f"Picked {len(picked)} handle vertices.")
    return np.array(picked, dtype=int)


def ask_handle_translation():
    """
    Ask for a translation vector (dx, dy, dz) for the selected handles.
    Example: 0 0.2 0
    """
    while True:
        try:
            txt = input("Enter dx dy dz for handle translation (e.g., 0 0.2 0): ")
            dx, dy, dz = map(float, txt.strip().split())
            return np.array([dx, dy, dz], dtype=np.float64)
        except Exception:
            print("Parse failed. Please try again. (e.g., 0 0.2 0)")


def visualize_deformation(faces, p_deformed, window_name="ARAP Deformed Mesh"):
    """
    Visualize the deformed mesh using Open3D.

    Args:
        faces: (F, 3) triangle indices
        p_deformed: (N, 3) deformed vertex positions
        window_name: title for the Open3D window
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(p_deformed)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh], window_name=window_name)
