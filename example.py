import numpy as np
import plotly.graph_objects as go

# 예시: V, F 로드
# V: (n, 3), F: (m, 3) 0-based 인덱스라고 가정
V = vertices  # np.ndarray
F = faces     # np.ndarray

mesh = go.Mesh3d(
    x=V[:, 0],
    y=V[:, 1],
    z=V[:, 2],
    i=F[:, 0],
    j=F[:, 1],
    k=F[:, 2],
    color='lightblue',
    opacity=1.0,
    flatshading=True,
)

fig = go.Figure(data=[mesh])
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)
fig.show()
