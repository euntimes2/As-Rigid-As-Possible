import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_laplacian(num_verts, w_ij):
    """
    num_verts: 정점 개수 (verts.shape[0])
    w_ij: list[dict], w_ij[i][j] = cotangent weight
    return: L (num_verts x num_verts) sparse matrix (CSC)
    """
    # LIL 포맷으로 만들면 row-wise로 채우기 편함
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

    # factorization 에 적합한 CSC 포맷으로 변환
    return L.tocsc()


def setup_constrained_system(L, fixed_idx, num_verts):
    """
    L: (N, N) sparse Laplacian
    fixed_idx: list/array of constrained vertex indices F
    num_verts: N
    return:
        free_idx: 자유 vertex 인덱스 배열 U
        L_UU: L[U, U]
        L_UF: L[U, F]
        solve_U: factorized solver for L_UU
    """
    fixed_idx = np.array(fixed_idx, dtype=int)
    all_idx = np.arange(num_verts, dtype=int)

    # U = 전체 - F
    free_idx = np.setdiff1d(all_idx, fixed_idx)

    # 부분 행렬들 뽑기 (sparse slicing)
    L_UU = L[free_idx][:, free_idx]   # (|U|, |U|)
    L_UF = L[free_idx][:, fixed_idx]  # (|U|, |F|)

    # factorization (한 번만)
    L_UU = L_UU.tocsc()
    solve_U = spla.factorized(L_UU)

    return free_idx, fixed_idx, L_UF, solve_U

def build_one_ring_neighbors(num_verts, faces):
    # 중복을 없애기 위해선 dictionary 대응을 set 으로 만듬
    neighbors = [set() for _ in range(num_verts)] 
    
    # faces 순환
    for f in faces:
        i, j, k = int(f[0]), int(f[1]), int(f[2])

        # 삼각형의 세 edge: (i, j), (j, k), (k, i)
        # 양방향으로 neighbor 추가
        neighbors[i].add(j)
        neighbors[j].add(i)

        neighbors[j].add(k)
        neighbors[k].add(j)

        neighbors[k].add(i)
        neighbors[i].add(k)

    # set 을 다시 list 로 변환
    return [list(nbs) for nbs in neighbors]

def compute_cotangent_weights(verts, faces):
    """
    verts: (N, 3) float32/float64
    faces: (M, 3) int32/int64
    return: edge_weight: list of dict, edge_weight[i][j] = weight_ij (symmetric)
    """
    num_verts = verts.shape[0]
    # 메모리 효율성을 위해서 sparse matrix 형태로 만들기
    w_ij = [dict() for _ in range(num_verts)]  # i -> {j: w_ij}

    def cotangent(a, b, c):
        """
        각도 at a, (b-a)와 (c-a) 사이의 cot(theta) = dot / ||cross||
        a, b, c: 3D points
        """
        u = b - a
        v = c - a
        cross = np.linalg.norm(np.cross(u, v))
        if cross < 1e-12:  # 거의 일직선인 경우
            return 0.0
        dot = np.dot(u, v)
        return dot / cross

    for f in faces:
        i, j, k = int(f[0]), int(f[1]), int(f[2])

        vi = verts[i]
        vj = verts[j]
        vk = verts[k]

        # edge (i, j) 를 기준으로 반대편에 존재하는 k 에서의 각도 구하기
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

def compute_b(verts, w_ij, R):
    """
    논문 식 (8)의 오른쪽 항 b 계산
    verts: (N, 3) original p
    neighbors: neighbors[i] = [j1, j2, ...]
    w_ij: w_ij[i][j] = cot weight
    R: (N, 3, 3) 각 vertex i에 대한 rotation matrix
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