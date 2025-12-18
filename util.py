import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_laplacian(num_verts, w_ij):
    """
    num_verts: number of vertices (verts.shape[0])
    w_ij: list[dict], w_ij[i][j] = cotangent weight
    return: L (num_verts x num_verts) sparse matrix (CSC)
    """
    # LIL format is convenient for row-wise assembly.
    L = sp.lil_matrix((num_verts, num_verts), dtype=np.float64)

    for i in range(num_verts):
        diag = 0.0
        for j, w in w_ij[i].items():
            if i == j:
                continue
            # off-diagonal
            L[i, j] -= w
            diag += w
        # diagonal
        L[i, i] = diag

    # Convert to CSC for factorization.
    return L.tocsc()


def setup_constrained_system(L, fixed_idx, num_verts):
    """
    L: (N, N) sparse Laplacian
    fixed_idx: list/array of constrained vertex indices F
    num_verts: N
    return:
        free_idx: free vertex indices U
        L_UU: L[U, U]
        L_UF: L[U, F]
        solve_U: factorized solver for L_UU
    """
    fixed_idx = np.array(fixed_idx, dtype=int)
    all_idx = np.arange(num_verts, dtype=int)

    # U = all - F
    free_idx = np.setdiff1d(all_idx, fixed_idx)

    # Extract submatrices (sparse slicing).
    L_UU = L[free_idx][:, free_idx]   # (|U|, |U|)
    L_UF = L[free_idx][:, fixed_idx]  # (|U|, |F|)

    # Factorization (once).
    L_UU = L_UU.tocsc()
    solve_U = spla.factorized(L_UU)

    return free_idx, fixed_idx, L_UF, solve_U

def build_one_ring_neighbors(num_verts, faces):
    # Use sets to avoid duplicates.
    neighbors = [set() for _ in range(num_verts)] 
    
    # Iterate faces.
    for f in faces:
        i, j, k = int(f[0]), int(f[1]), int(f[2])

        # Triangle edges: (i, j), (j, k), (k, i)
        # Add neighbors bidirectionally.
        neighbors[i].add(j)
        neighbors[j].add(i)

        neighbors[j].add(k)
        neighbors[k].add(j)

        neighbors[k].add(i)
        neighbors[i].add(k)

    # Convert sets back to lists.
    return [sorted(list(nbs)) for nbs in neighbors]

def compute_cotangent_weights(verts, faces):
    """
    verts: (N, 3) float32/float64
    faces: (M, 3) int32/int64
    return: edge_weight: list of dict, edge_weight[i][j] = weight_ij (symmetric)
    """
    num_verts = verts.shape[0]
    # Use list-of-dicts for memory efficiency.
    w_ij = [dict() for _ in range(num_verts)]  # i -> {j: w_ij}

    def cotangent(a, b, c):
        """
        Angle at a, cot(theta) between (b-a) and (c-a) = dot / ||cross||
        a, b, c: 3D points
        """
        u = b - a
        v = c - a
        cross = np.linalg.norm(np.cross(u, v))
        if cross < 1e-12:  # Nearly collinear.
            return 0.0
        dot = np.dot(u, v)
        return dot / cross

    for f in faces:
        i, j, k = int(f[0]), int(f[1]), int(f[2])

        vi = verts[i]
        vj = verts[j]
        vk = verts[k]

        # Angle at k opposite edge (i, j).
        cot_k = cotangent(vk, vi, vj)
        w_ij[i][j] = w_ij[i].get(j, 0.0) + 0.5 * cot_k
        w_ij[j][i] = w_ij[j].get(i, 0.0) + 0.5 * cot_k
    
        # edge (j, k) opposite i → angle at i
        cot_i = cotangent(vi, vj, vk)
        w_ij[j][k] = w_ij[j].get(k, 0.0) + 0.5 * cot_i
        w_ij[k][j] = w_ij[k].get(j, 0.0) + 0.5 * cot_i

        # edge (k, i) opposite j → angle at j
        cot_j = cotangent(vj, vk, vi)
        w_ij[k][i] = w_ij[k].get(i, 0.0) + 0.5 * cot_j
        w_ij[i][k] = w_ij[i].get(k, 0.0) + 0.5 * cot_j

    return w_ij

def compute_uniform_weights(neighbors):
    """
    uniform weight:
      인접한 정점 i-j에 대해 w_ij = 1 (대칭)
    반환 형식은 compute_cotangent_weights와 동일하게
      w_ij[i] = {j: weight, ...} 형태라고 가정
    """
    num_verts = len(neighbors)
    w_ij = [dict() for _ in range(num_verts)]
    for i in range(num_verts):
        for j in neighbors[i]:
            w_ij[i][j] = 1.0
            w_ij[j][i] = 1.0
    return w_ij

def compute_b(verts, w_ij, R):
    """
    Compute the right-hand side b from Eq. (8).
    verts: (N, 3) original p
    neighbors: neighbors[i] = [j1, j2, ...]
    w_ij: w_ij[i][j] = cot weight
    R: (N, 3, 3) rotation matrix per vertex i
    return: b (N, 3)
    """
    num_verts = verts.shape[0]
    b = np.zeros((num_verts, 3), dtype=np.float64)

    for i in range(num_verts):
        bi = np.zeros(3, dtype=np.float64)
        pi = verts[i]

        for j, w in w_ij[i].items():
            pj = verts[j]
            pij = pi - pj  # (p_i - p_j)

            Ri = R[i]
            Rj = R[j]

            # 0.5 * (R_i + R_j) @ (p_i - p_j)
            term = 0.5 * (Ri + Rj) @ pij

            bi += w * term

        b[i] = bi

    return b
