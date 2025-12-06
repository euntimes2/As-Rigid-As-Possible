# visualization.py

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


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


def visualize_deformation(verts, faces, p_deformed, html_path="arap_deformation.html", open_browser=True):
    """
    Plotly 기반의 3D 웹 뷰어로 변형된 mesh 를 시각화한다.

    - 리눅스/WSL 에서도 HTML 파일을 브라우저로 열 수 있도록
      ``html_path`` 로 저장한다.
    - ``open_browser`` 가 True 이면 plotly 가 기본 브라우저를 자동으로 띄운다.

    Args:
        verts: (N, 3) 원래 vertex 위치 (현재 함수에서는 범위 계산용)
        faces: (F, 3) 삼각형 face index 배열
        p_deformed: (N, 3) 변형된 vertex 위치
        html_path: 저장할 HTML 파일 경로 (기본: ``arap_deformation.html``)
        open_browser: True 인 경우 저장 후 브라우저 자동 오픈
    """

    x, y, z = p_deformed.T
    i, j, k = faces.T

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color="lightblue",
        opacity=0.85,
        flatshading=True,
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.7, roughness=0.7, specular=0.25),
        lightposition=dict(x=0.5, y=0.5, z=1.0),
    )

    fig = go.Figure(mesh)
    fig.update_layout(
        title="ARAP Deformed Mesh (Plotly)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    output_path = Path(html_path).resolve()
    pio.write_html(fig, file=output_path, auto_open=open_browser, include_plotlyjs="cdn")
    print(f"Plotly 시각화가 {output_path} 에 저장되었어. 브라우저에서 열어봐!")
