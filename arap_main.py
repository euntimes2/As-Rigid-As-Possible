
from data_ply import load_ply
from util import compute_b, setup_constrained_system, build_one_ring_neighbors, compute_cotangent_weights, build_laplacian
from visualization import pick_handles, ask_handle_translation, visualize_deformation

import scipy.sparse.linalg as spla
import numpy as np


# 6. Rotation matrix: vertex 당 대응되는 rotation matrix 구하기
# 6-1. edge matrix after deformation 구하기 e'_ij = p'_i - p'_j
# 6-2. S_i = P_i * D_i * P'_i^T
# 6-3. S_i SVD 분해 = U_i, sigma_i , V_i^TT
# 6-4. R_i = V_i*U_i^T
def compute_rotations(verts, p_deformed, neighbors, w_ij, eps=1e-12):
    """
    ARAP Local Step:
    각 vertex i에 대해, one-ring 이웃과 cotangent weight를 이용해
    최적의 rotation matrix R_i 를 SVD로 계산.

    verts:      (N, 3) 원래 vertex 위치 p_i
    p_deformed: (N, 3) 변형된 vertex 위치 p'_i
    neighbors:  list of list, neighbors[i] = [j1, j2, ...]
    w_ij:       list of dict, w_ij[i][j] = scalar weight
    eps:        degenerate 방지용 작은 값
    return:
        R: (N, 3, 3) numpy array, 각 i에 대응하는 회전행렬 R_i
    """
    num_verts = verts.shape[0]
    R = np.zeros((num_verts, 3, 3), dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    for i in range(num_verts):
        # connectivity 가 끊어지지 않기 때문에 neighbors 는 deformation 전후 동일함
        js = neighbors[i]
        if len(js) == 0:
            # 이웃 없는 고립된 점이면 identity
            R[i] = I
            continue

        # original / deformed center
        pi_orig = verts[i]        
        pi_def  = p_deformed[i]   

        # one-ring edge들과 weight 모으기
        e_list   = []
        e_def_list = []
        w_list   = []
        # i 에 대한 neighbor 순회
        for j in js:
            pj_orig = verts[j]
            pj_def  = p_deformed[j]

            # original edge: e_ij = p_i - p_j
            e_ij = pi_orig - pj_orig
            # deformed edge: e'_ij = p'_i - p'_j
            e_ij_def = pi_def - pj_def

            w = w_ij[i].get(j, 0.0)

            e_list.append(e_ij)
            e_def_list.append(e_ij_def)
            w_list.append(w)

        P_i = np.stack(e_list, axis=1)           
        P_def_i = np.stack(e_def_list, axis=1)  
        W_i = np.diag(w_list) 
        # 6-2. S_i = P_i * D_i * P'_i^T
        S_i = P_i @ W_i @ P_def_i.T     

        # 너무 degenerate한 경우(거의 0 matrix)이면 identity 사용
        if np.linalg.norm(S_i) < eps:
            R[i] = I
            continue

        # 6-3. S_i SVD 분해 = U_i, sigma_i , V_i^TT
        U, _, Vt = np.linalg.svd(S_i)

        # 6-4. R_i = V_i*U_i^T
        Ri = Vt.T @ U.T

        # det(R) < 0 이면 reflection → 마지막 축 뒤집어서 rotation으로 보정
        if np.linalg.det(Ri) < 0:
            Vt[-1, :] *= -1
            Ri = Vt.T @ U.T

        R[i] = Ri

    return R


def arap_global_step_constrained(
    verts,
    w_ij,
    R,
    free_idx,
    fixed_idx,
    L_UF,
    solve_U,
    fixed_positions
):
    """
    ARAP Global Step:
    L p' = b 를 좌표별로 풀어서 새로운 p'를 계산

    verts: (N, 3) original p
    neighbors, w_ij, R: 위와 동일
    solve_L: scipy.sparse.linalg.factorized(L) 로 만든 solver 함수
    return: p_new (N, 3) 업데이트된 변형 vertex 위치
    """

    b = compute_b(verts, w_ij, R)  # (N, 3)

    # 2. U/F 나눔
    b_U = b[free_idx, :]     # (|U|, 3)
    c_F = fixed_positions         # (|F|, 3)

    LUF_c = L_UF @ c_F            # (|U|, 3)
    b_tilde = b_U - LUF_c         # (|U|, 3)

    num_verts = verts.shape[0]
    p_new = np.zeros((num_verts, 3), dtype=np.float64)
    p_new[fixed_idx, :] = c_F

    # 각 좌표별로 L p' = b_x, b_y, b_z 풀기
    for k in range(3):
        p_new[free_idx, k] = solve_U(b_tilde[:, k])

    return p_new



def main():
    # 1. data 불러오기
    data_path = './data/armadillo_simplified.ply'
    verts, faces = load_ply(data_path)

    # 2. data 에 대해서 cell 로 나누기
    num_verts = verts.shape[0]
    neighbors = build_one_ring_neighbors(num_verts, faces)

    # 3. edge weight matrix 계산하기: cotn weight
    edge_weights = compute_cotangent_weights(verts, faces)

    # 4. laplacian L 만들기
    L = build_laplacian(num_verts, edge_weights)

    # 5. Open3D 로 handle vertex 선택
    handle_idx = pick_handles(verts, faces)
    if len(handle_idx) == 0:
        print("Handle vertex 를 하나도 안 골라서, 그냥 종료할게.")
        return
    
    # 6. handle 들을 얼마나 옮길지 입력
    delta = ask_handle_translation()
    fixed_positions = verts[handle_idx] + delta  # p'_F = p_F + delta
    handle_idx = np.array(handle_idx, dtype=int)

    # while 종료 신호 들어올 때까지
    # 5. input 으로 움직일 vertex + vertex 의 움직인 후 위치 받아오기
    # 5-1. 이 때, 움직일 vertex 에 대해 vertex 가 움직인 후의 위치 c 는 fixed 가정 -> optimization 대상 아님 
    # 4. 사용자 constraint 정의 (예: vertex 0, 10, 200 을 고정)
    # 5. L : Factoization pre calculate with edge weight matrix
    # -> input 에 의해서 L matrix 에서 fix 된 대응되는 row 가 소거됨
    free_idx , fixed_idx, L_UF, solve_L = setup_constrained_system(L, handle_idx, num_verts)
    
    p_deformed = verts.copy()
    p_deformed[fixed_idx] = fixed_positions

    # while convergence
    max_iter = 100

    for it in range(max_iter):
        # Local step: R_i 계산
        R = compute_rotations(verts, p_deformed, neighbors, edge_weights)

        # Global step: (constraint-aware)
        p_new = arap_global_step_constrained(
            verts=verts,
            w_ij=edge_weights,
            R=R,
            free_idx=free_idx,
            fixed_idx=fixed_idx,
            L_UF=L_UF,
            solve_U=solve_L,
            fixed_positions=fixed_positions,
        )

        # 수렴 체크 (optional)
        diff = np.linalg.norm(p_new - p_deformed)
        p_deformed = p_new
        print(f"Optimizing in step {it}, diff = {diff}")
        if diff < 1e-6:
            print(f"Optimizing in step {it}, diff = {diff}")
            break
     # 10. 결과 시각화
    visualize_deformation(verts, faces, p_deformed)
  
    


if __name__ == "__main__":
    
    main()
