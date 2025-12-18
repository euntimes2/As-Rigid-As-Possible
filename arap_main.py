from data_ply import load_ply
from util import (
    compute_b,
    setup_constrained_system,
    build_one_ring_neighbors,
    compute_cotangent_weights,
    compute_uniform_weights,
    build_laplacian,
)
from visualization import pick_handles, ask_handle_translation, visualize_deformation

import scipy.sparse.linalg as spla
import numpy as np
import argparse

# -----------------------------
# ARAP Local Step: 각 정점 i에 대해 최적 회전 R_i를 SVD로 구한다.
# -----------------------------
# 핵심 아이디어:
#   원본 1-ring 에지 벡터들 (p_i - p_j) 를
#   변형 후 1-ring 에지 벡터들 (p'_i - p'_j) 에
#   가장 잘 맞추는(least squares) 회전 R_i를 찾는다.
#
# 수식(대표 형태):
#   S_i = Σ_j w_ij (p'_i - p'_j) (p_i - p_j)^T
#   SVD(S_i) = U Σ V^T
#   R_i = U V^T   (det<0 이면 반사(reflection) 보정)
def compute_rotations(verts, p_deformed, neighbors, w_ij, eps=1e-12):
    """
    ARAP 로컬 단계(Local step):
    각 정점 i의 최적 회전 행렬 R_i를 1-ring 이웃과 cotangent weight로 추정한다.

    verts:      (N, 3) 원본 정점 좌표 p_i
    p_deformed: (N, 3) 현재 변형 정점 좌표 p'_i
    neighbors:  list[list[int]] 형태의 1-ring 이웃 리스트
    w_ij:       list[dict] 형태의 가중치; w_ij[i][j] = cotangent weight
    eps:        수치적으로 너무 작은 경우(퇴화) 방지용
    return:
        R: (N, 3, 3) 각 정점 i에 대한 회전 행렬 R_i
    """
    num_verts = verts.shape[0]
    R = np.zeros((num_verts, 3, 3), dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    for i in range(num_verts):
        js = neighbors[i]
        if len(js) == 0:
            # 이웃이 없는 고립 정점은 회전을 정의할 수 없으니 항등행렬로 둔다.
            R[i] = I
            continue

        # 정점 i의 원본 / 현재 변형 위치
        pi_orig = verts[i]
        pi_def = p_deformed[i]

        # i의 1-ring 이웃에 대해 에지 벡터와 가중치를 모은다.
        #   e_ij     = p_i  - p_j
        #   e'_ij    = p'_i - p'_j
        e_list = []
        e_def_list = []
        w_list = []

        for j in js:
            pj_orig = verts[j]
            pj_def = p_deformed[j]

            e_ij = pi_orig - pj_orig
            e_ij_def = pi_def - pj_def

            w = w_ij[i].get(j, 0.0)

            e_list.append(e_ij)
            e_def_list.append(e_ij_def)
            w_list.append(w)

        # 행렬로 쌓기: (3, deg(i))
        P_i = np.stack(e_list, axis=1)
        P_def_i = np.stack(e_def_list, axis=1)

        # 가중치 대각행렬: (deg(i), deg(i))
        W_i = np.diag(w_list)

        # 공분산(상관) 행렬 S_i 구성:
        #   S_i = Σ_j w_ij (e'_ij)(e_ij)^T
        #   = P'_i W_i P_i^T
        S_i = P_def_i @ W_i @ P_i.T

        # 너무 작은 경우(퇴화/수치 문제)는 항등 회전
        if np.linalg.norm(S_i) < eps:
            R[i] = I
            continue

        # SVD: S_i = U Σ V^T
        U, _, Vt = np.linalg.svd(S_i)

        # 최적 회전: R_i = U V^T
        Ri = U @ Vt

        # det(R_i)<0 이면 반사(reflection)이 섞인 것이므로 보정
        if np.linalg.det(Ri) < 0:
            Vt[-1, :] *= -1
            Ri = U @ Vt

        R[i] = Ri

    return R


# -----------------------------
# ARAP Global Step: R_i들을 고정한 채로 p'를 선형 시스템으로 푼다(제약 포함).
# -----------------------------
# 핵심 아이디어:
#   로컬에서 얻은 회전 R_i를 사용해 우변 b를 만들고,
#   Laplacian 기반 선형 시스템을 풀어 변형 정점 p'를 업데이트한다.
#   또한 handle/anchor 같은 고정 정점(Fixed set)은 지정된 target 위치로 강제한다.
def arap_global_step_constrained(
    verts,
    w_ij,
    R,
    free_idx,
    fixed_idx,
    L_UF,
    solve_U,
    fixed_positions,
):
    """
    ARAP 글로벌 단계(Global step, constraint-aware):
    고정된 정점(F)과 자유 정점(U)으로 분할된 시스템에서
    자유 정점 위치 p'_U를 풀고, 고정 정점은 target으로 둔다.

    verts:           (N, 3) 원본 정점 좌표 p
    w_ij:            cotangent weights
    R:               (N, 3, 3) 로컬 단계에서 얻은 회전들
    free_idx:        자유 정점 인덱스(U)
    fixed_idx:       고정 정점 인덱스(F)  (handles + anchors)
    L_UF:            분할 라플라시안 블록( U행, F열 ) 형태
    solve_U:         L_UU에 대한 factorized solver (solve_U(x)로 풂)
    fixed_positions: (|F|, 3) 고정 정점들의 목표 위치 p'_F
    return:
        p_new: (N, 3) 업데이트된 변형 정점 좌표 p'
    """

    # 우변 b 계산 (util.compute_b가 ARAP 수식에 맞게 구현되어 있어야 함)
    b = compute_b(verts, w_ij, R)  # (N, 3)

    # U/F 분할
    b_U = b[free_idx, :]          # (|U|, 3)
    c_F = fixed_positions         # (|F|, 3)

    # U에 대한 유효 우변: b~ = b_U - L_UF * c_F
    LUF_c = L_UF @ c_F
    b_tilde = b_U - LUF_c

    num_verts = verts.shape[0]
    p_new = np.zeros((num_verts, 3), dtype=np.float64)

    # 고정 정점은 target으로 박아둔다.
    p_new[fixed_idx, :] = c_F

    # 좌표별로 선형 시스템 풀기 (x,y,z 각각)
    for k in range(3):
        p_new[free_idx, k] = solve_U(b_tilde[:, k])

    return p_new


# -----------------------------
# Anchor 선택: 전체가 통째로 떠다니는 강체 모드 제거를 위해 바닥 일부를 고정한다.
# -----------------------------
def choose_anchor_vertices(verts, anchor_percent=0.02):
    """
    y좌표가 가장 낮은 하위 anchor_percent 비율의 정점들을 앵커로 선택한다.
    (예: 2%이면 바닥 근처 정점들을 고정하여 전체 translation/rotation이 생기는 걸 막음)
    """
    y = verts[:, 1]
    thresh = np.quantile(y, anchor_percent)
    anchor_idx = np.where(y <= thresh)[0]
    if anchor_idx.size == 0:
        anchor_idx = np.array([int(np.argmin(y))], dtype=int)
    return anchor_idx.astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="cot", choices=["cot", "uniform"],
                        help="edge weights: cot (cotangent) or uniform")
    args = parser.parse_args()
    # 1) 메쉬 로드
    data_path = "./data/armadillo_simplified.ply"
    verts, faces = load_ply(data_path)

    # 2) 1-ring 이웃(정점 인접 리스트) 구성
    num_verts = verts.shape[0]
    neighbors = build_one_ring_neighbors(num_verts, faces)

    # 3) cotangent 가중치 계산 (edge weights)
    if args.weight == "cot":
        print("[Weight] Using cotangent weights")
        edge_weights = compute_cotangent_weights(verts, faces)
    else:
        print("[Weight] Using uniform weights (w_ij = 1)")
        edge_weights = compute_uniform_weights(neighbors)

    # 4) 라플라시안 L 구성
    L = build_laplacian(num_verts, edge_weights)

    # 5) Open3D로 핸들 정점 선택 (박스 드래그로 선택)
    handle_idx = pick_handles(verts, faces)
    if len(handle_idx) == 0:
        print("핸들을 선택하지 않아서 종료합니다.")
        return

    # 6) 핸들에 적용할 이동량(translation) 입력
    delta = ask_handle_translation()

    # handle 인덱스를 numpy로 정리
    handle_idx = np.array(handle_idx, dtype=int)

    # 7) 앵커 추가: 바닥 일부 정점 고정
    anchor_idx = choose_anchor_vertices(verts, anchor_percent=0.02)
    print(f"[Anchor] Anchoring {len(anchor_idx)} vertices (lowest y 2%).")

    # 8) 고정 정점들의 target 위치 구성
    #    - handle: verts + delta 로 이동
    #    - anchor: 원래 위치 유지
    handle_targets = verts[handle_idx] + delta
    anchor_targets = verts[anchor_idx].copy()

    # 고정 정점 집합(F) = handles + anchors
    fixed_idx_input = np.concatenate([handle_idx, anchor_idx], axis=0)

    # 9) 제약 시스템 분할(한 번만 준비): fixed를 제거한 L_UU 시스템을 factorize
    #    주의: 내부에서 fixed_idx를 정렬/unique 할 수 있으므로,
    #    fixed_positions도 최종 fixed_idx 순서에 맞게 재구성해야 안전함.
    free_idx, fixed_idx, L_UF, solve_L = setup_constrained_system(L, fixed_idx_input, num_verts)

    # fixed_idx 최종 순서에 맞춰 target 매칭
    target_map = {int(i): handle_targets[k] for k, i in enumerate(handle_idx)}
    target_map.update({int(i): anchor_targets[k] for k, i in enumerate(anchor_idx)})
    fixed_positions = np.stack([target_map[int(i)] for i in fixed_idx], axis=0)

    # -----------------------------
    # 10) 초기화: Laplacian Editing 형태로 한 번 global step을 풀어서 p' 초기값을 만든다.
    #     (R = I 로 두고 global step을 수행)
    # -----------------------------
    R0 = np.tile(np.eye(3, dtype=np.float64)[None, :, :], (num_verts, 1, 1))
    p_deformed = arap_global_step_constrained(
        verts=verts,
        w_ij=edge_weights,
        R=R0,
        free_idx=free_idx,
        fixed_idx=fixed_idx,
        L_UF=L_UF,
        solve_U=solve_L,
        fixed_positions=fixed_positions,
    )

    # 11) 초기 변형 결과(ARAP 반복 전) 시각화
    print("[VIS] Laplacian-init deformation (before ARAP refinement)")
    # !!! 주의: visualize_deformation 시그니처에 맞게 호출해야 함
    # 보통 visualize_deformation(verts, faces, p_deformed) 형태가 정상
    visualize_deformation(faces, p_deformed)

    # 12) ARAP Local-Global 반복
    max_iter = 10
    for it in range(max_iter):
        # (Local) 회전 R_i 업데이트
        R = compute_rotations(verts, p_deformed, neighbors, edge_weights)

        # (Global) 정점 위치 p' 업데이트 (제약 포함)
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

        # 수렴 체크
        diff = np.linalg.norm(p_new - p_deformed)
        p_deformed = p_new
        print(f"[ARAP] iter {it:02d} | diff = {diff:.6e}")

        if diff < 1e-6:
            print(f"[ARAP] converged at iter {it:02d} | diff = {diff:.6e}")
            break

    # 13) 최종 결과 시각화
    visualize_deformation(faces, p_deformed)


if __name__ == "__main__":
    main()
