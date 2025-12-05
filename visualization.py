# visualization.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def pick_handles(verts, faces):
    """
    Open3D 대신, 콘솔에서 handle vertex 인덱스를 직접 입력받는 버전.

    예)
        선택할 vertex index 들을 공백으로 구분해서 입력해줘 (예: 10 42 180):
        10 42 180
    """
    print(f"전체 vertex 개수: {verts.shape[0]}")
    print("선택할 vertex index 들을 공백으로 구분해서 입력해줘 (예: 10 42 180):")

    while True:
        txt = input("handle vertex indices: ").strip()
        try:
            if not txt:
                print("아무 것도 입력 안 했어. 최소 하나는 골라줘.")
                continue
            idx_list = [int(x) for x in txt.split()]
            # 범위 체크
            for idx in idx_list:
                if idx < 0 or idx >= verts.shape[0]:
                    raise ValueError(f"vertex index {idx} 가 범위를 벗어났어 (0 ~ {verts.shape[0]-1})")
            print("Picked handle indices:", idx_list)
            return np.array(idx_list, dtype=int)
        except Exception as e:
            print("파싱 실패:", e)
            print("다시 입력해줘. 예: 0 10 200")


def ask_handle_translation():
    """
    콘솔에서 dx, dy, dz 입력받기.
    예: 0 0.2 0 이런 식.
    """
    import numpy as np
    while True:
        try:
            txt = input("Handle vertex 들을 얼마나 움직일지 dx dy dz 를 입력해줘 (예: 0 0.2 0): ")
            dx, dy, dz = map(float, txt.strip().split())
            return np.array([dx, dy, dz], dtype=np.float64)
        except Exception as e:
            print("파싱 실패. 다시 입력해줘. (예: 0 0.2 0)")


def visualize_deformation(verts, faces, p_deformed):
    """
    matplotlib 의 3D plot 으로 변형된 mesh 를 시각화.
    원본/변형 둘 다 보고 싶으면 subplot 두 개로 그려도 되고,
    여기서는 변형 mesh만 그려줄게.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # faces: (F, 3) int, p_deformed: (N, 3)
    tris = p_deformed[faces]  # (F, 3, 3)

    mesh_collection = Poly3DCollection(tris, alpha=0.8)
    mesh_collection.set_edgecolor("k")
    ax.add_collection3d(mesh_collection)

    # 축 스케일 맞추기
    all_pts = p_deformed
    x_min, y_min, z_min = all_pts.min(axis=0)
    x_max, y_max, z_max = all_pts.max(axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("ARAP Deformed Mesh (matplotlib)")

    plt.tight_layout()
    plt.show()
