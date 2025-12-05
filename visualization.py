import open3d as o3d
import numpy as np

def create_mesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh

def pick_handles(verts, faces):
    """
    Open3D 창 띄워서 handle로 쓸 vertex 인덱스 선택.
    Shift + Left Click으로 여러 개 찍고, Q/Esc로 종료.
    """
    mesh = create_mesh(verts, faces)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick handle vertices (Shift+LeftClick, Q to finish)")
    vis.add_geometry(mesh)
    print("Shift + Left Click 으로 vertex 찍고, Q/Esc 로 종료해줘.")
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    print("Picked handle indices:", picked)
    return picked
def ask_handle_translation():
    """
    콘솔에서 dx, dy, dz 입력받기.
    예: 0 0.2 0 이런 식.
    """
    while True:
        try:
            txt = input("Handle vertex 들을 얼마나 움직일지 dx dy dz 를 입력해줘 (예: 0 0.2 0): ")
            dx, dy, dz = map(float, txt.strip().split())
            return np.array([dx, dy, dz], dtype=np.float64)
        except Exception as e:
            print("파싱 실패. 다시 입력해줘. (예: 0 0.2 0)")

def visualize_deformation(verts, faces, p_deformed):
    """
    원본과 변형 mesh 둘 다 보고 싶으면 두 개 띄우거나,
    여기서는 변형 mesh만 보여주는 걸로.
    """
    mesh_def = create_mesh(p_deformed, faces)
    o3d.visualization.draw_geometries([mesh_def], window_name="ARAP Deformed Mesh")
