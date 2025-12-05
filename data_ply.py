import trimesh
import numpy as np

def load_ply(path):
    mesh = trimesh.load(path, process=False)
    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    return verts, faces