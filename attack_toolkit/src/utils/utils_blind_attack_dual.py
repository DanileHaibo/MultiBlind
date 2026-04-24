"""Helpers for dual blind-spot (two lens flares) attacks.

Kept separate from utils_blind_attack.py so single-blind behavior stays unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import LineString, Point


def _curvature_2d(points: np.ndarray) -> np.ndarray:
    """与 ``utils_attack.calculate_curvature`` 同式，纯 numpy，避免本模块顶层依赖 mmdet3d 链。"""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curv = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2) ** 1.5
    curv[np.isinf(curv)] = 0.0
    return np.nan_to_num(curv, nan=0.0)


def pick_two_curve_anchor_xy(
    diverge_boundary_pts,
    reference_boundary_pts,
    min_sep_m=3.0,
    curvature_diff_threshold=0.1,
    top_k=5,
):
    """Pick two 2D anchor points on diverge boundary that correspond to distinct curve
    regions (asymmetry vs reference), favoring large separation along the road.

    Returns:
        (anchor_a_xy, anchor_b_xy): each shape (2,)
    """
    from attack_toolkit.src.utils.utils_attack import get_asymmetry_anchors

    anchors = get_asymmetry_anchors(
        diverge_boundary_pts,
        reference_boundary_pts,
        CURVATURE_DIFF_THRESHOLD=curvature_diff_threshold,
        top_k=top_k,
    )
    anchors_xy = np.asarray(anchors)[:, :2]

    if len(anchors_xy) < 2:
        # Degenerate: split boundary into two arc positions
        line = LineString(np.asarray(diverge_boundary_pts))
        L = line.length
        if L < 1e-3:
            p0 = np.asarray(diverge_boundary_pts[0], dtype=np.float64)
            p1 = p0 + np.array([min_sep_m, 0.0])
            return p0, p1
        p_a = np.array(line.interpolate(L * 0.25).coords[0], dtype=np.float64)
        p_b = np.array(line.interpolate(L * 0.75).coords[0], dtype=np.float64)
        return p_a, p_b

    best = None
    best_sep = -1.0
    n = len(anchors_xy)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(anchors_xy[i] - anchors_xy[j]))
            if d >= min_sep_m and d > best_sep:
                best_sep = d
                best = (anchors_xy[i].copy(), anchors_xy[j].copy())
    if best is not None:
        return best[0], best[1]

    # Relax separation: take farthest pair anyway
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(anchors_xy[i] - anchors_xy[j]))
            if d > best_sep:
                best_sep = d
                best = (anchors_xy[i].copy(), anchors_xy[j].copy())
    return best[0], best[1]


def build_dual_pools_by_anchor_proximity(
    attack_loc_candidates_sorted,
    anchor_a_xy,
    anchor_b_xy,
    pool_per_anchor,
):
    """Split ranked (point, score) candidates into two pools by proximity to each anchor.

    attack_loc_candidates_sorted: list of (attack_loc_3d, score), sorted by score descending.
    """
    cand = attack_loc_candidates_sorted
    with_da = [(p, s, np.linalg.norm(p[:2] - anchor_a_xy)) for p, s in cand]
    with_db = [(p, s, np.linalg.norm(p[:2] - anchor_b_xy)) for p, s in cand]
    pool_a = [p for p, s, _ in sorted(with_da, key=lambda x: x[2])[:pool_per_anchor]]
    pool_b = [p for p, s, _ in sorted(with_db, key=lambda x: x[2])[:pool_per_anchor]]
    return pool_a, pool_b


def apply_dual_lens_flare(
    imgs,
    img_metas,
    img_norm_cfg,
    position_1,
    position_2,
    ori_hw,
):
    """Apply two sequential lens flares (same physics as single-blind, stacked)."""
    from attack_toolkit.src.utils.utils_blind_attack import generate_lens_flare

    ori_h, ori_w = ori_hw
    light1 = {
        'position': np.asarray(position_1, dtype=np.float64),
        'power': 3000.0,
        'beam_angle': np.radians(40),
    }
    light2 = {
        'position': np.asarray(position_2, dtype=np.float64),
        'power': 3000.0,
        'beam_angle': np.radians(40),
    }
    for cam_idx in range(6):
        imgs = generate_lens_flare(
            imgs, img_metas, light1, img_norm_cfg, cam_idx, (ori_h, ori_w)
        )
    for cam_idx in range(6):
        imgs = generate_lens_flare(
            imgs, img_metas, light2, img_norm_cfg, cam_idx, (ori_h, ori_w)
        )
    return imgs


def iter_dual_location_pairs(pool_a, pool_b, min_pair_sep_m=1.5):
    """Yield (loc1, loc2) with loc1 from pool_a, loc2 from pool_b, not too close."""
    for loc1 in pool_a:
        for loc2 in pool_b:
            if np.linalg.norm(loc1[:2] - loc2[:2]) < min_pair_sep_m:
                continue
            yield loc1, loc2


def nearest_anchor_index(xy, anchor_a_xy, anchor_b_xy):
    """Return 0 if closer to anchor A, else 1 (closer to B)."""
    da = float(np.linalg.norm(xy - anchor_a_xy))
    db = float(np.linalg.norm(xy - anchor_b_xy))
    return 0 if da <= db else 1


def build_second_curve_search_list(
    attack_loc_candidates,
    anchor_a_xy,
    anchor_b_xy,
    loc1,
    total_locs,
    min_pair_sep_m,
):
    """[Legacy] 第二眩光在 diverge 上、相对 loc1 的另一曲率锚侧。已由
    :func:`build_p2_on_reference_boundary` 取代（P2 改在对面 reference 线上搜）。"""
    ia = nearest_anchor_index(np.asarray(loc1)[:2], anchor_a_xy, anchor_b_xy)

    def far_enough(p):
        return np.linalg.norm(p[:2] - loc1[:2]) >= min_pair_sep_m

    tier1 = [
        (p, s)
        for p, s in attack_loc_candidates
        if far_enough(p) and nearest_anchor_index(p[:2], anchor_a_xy, anchor_b_xy) != ia
    ]
    tier1.sort(key=lambda x: x[1], reverse=True)
    pts = [p for p, _ in tier1[:total_locs]]
    if len(pts) > 0:
        return pts

    tier2 = [(p, s) for p, s in attack_loc_candidates if far_enough(p)]
    tier2.sort(key=lambda x: x[1], reverse=True)
    pts = [p for p, _ in tier2[:total_locs]]
    if len(pts) > 0:
        return pts

    # Last resort: farthest points from loc1 among top-scored candidates
    ranked = sorted(attack_loc_candidates, key=lambda x: x[1], reverse=True)
    with_d = [(p, s, np.linalg.norm(p[:2] - loc1[:2])) for p, s in ranked[: max(200, total_locs)]]
    with_d.sort(key=lambda x: x[2], reverse=True)
    return [p for p, _, _ in with_d[:total_locs]]


def build_opposite_curved_lane_polyline(
    diverge_boundary_pts,
    reference_boundary_pts,
    n_diverge_samples=400,
    min_segment_len_m=0.5,
):
    """在 reference 上得到与 P1 所在弯段**对应**的「对边同弯」车道/路沿折线。

    在 diverge 上按弧长匀采样，将各点正交到 reference 上取最近点，在 reference
    弧长上得到 ``[s_min, s_max]``，再裁出该子折线并重采样顶点。P2 只应在这条
    对边**弯道对应段**上搜索，而不要用整条 reference（可能含长直段、与 P1
    纵向上不对齐）。

    若子段过短，退回整条 reference 以保证候选非空。

    Returns:
        np.ndarray, shape (M, 2), 2D 折线 (x, y) in lidar.
    """
    div = np.asarray(diverge_boundary_pts, dtype=np.float64)
    ref = np.asarray(reference_boundary_pts, dtype=np.float64)
    if div.shape[0] < 2 or ref.shape[0] < 2:
        return ref
    d_line = LineString(div)
    r_line = LineString(ref)
    ld, lr = d_line.length, r_line.length
    if ld < 1e-8 or lr < 1e-8:
        return ref

    n = int(max(20, n_diverge_samples))
    proj = []
    for k in range(n):
        t = (k + 0.5) * ld / n
        p = d_line.interpolate(t)
        proj.append(float(r_line.project(Point(p.x, p.y))))
    s0, s1 = float(min(proj)), float(max(proj))
    s0 = max(0.0, min(s0, lr))
    s1 = max(0.0, min(s1, lr))
    if s1 < s0 + 1e-3 or s1 - s0 < min_segment_len_m * 0.1:
        s0, s1 = 0.0, lr
    n_vert = max(2, int((s1 - s0) / 0.25) + 1)
    s_grid = np.linspace(s0, s1, n_vert, dtype=np.float64)
    out = []
    for s in s_grid:
        c = r_line.interpolate(s)
        out.append([c.x, c.y])
    return np.asarray(out, dtype=np.float64)


def _tensor_list_to_line_xy(fixed_pts, idx: int):
    if fixed_pts is None or idx < 0 or idx >= len(fixed_pts):
        return None
    bb = fixed_pts[idx]
    if hasattr(bb, 'detach'):
        arr = bb.detach().cpu().numpy()
    elif hasattr(bb, 'numpy'):
        arr = bb.numpy()
    else:
        arr = np.asarray(bb)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    return np.asarray(arr[:, :2], dtype=np.float64)


def _road_lateral_frame(left_boundary_pts, right_boundary_pts):
    """左右路沿均值定义横向：lat_hat 由左指向右。mid 为两路沿中点。"""
    l_poly = np.asarray(left_boundary_pts, dtype=np.float64)[:, :2]
    r_poly = np.asarray(right_boundary_pts, dtype=np.float64)[:, :2]
    l_mid = l_poly.mean(axis=0)
    r_mid = r_poly.mean(axis=0)
    mid = 0.5 * (l_mid + r_mid)
    lat = r_mid - l_mid
    nl = float(np.linalg.norm(lat))
    if nl < 1e-6:
        return None
    return mid, lat / nl


def _on_same_side_as_diverge_tag(
    c_xy: np.ndarray,
    diverge_tag: str,
    mid: np.ndarray,
    lat_hat: np.ndarray,
) -> bool:
    """
    与 P1 同侧：P1 在左沿→左半幅 ``dot(p-mid,lat)<0``；P1 在右沿→右半幅 ``dot>0``。
    """
    d = float(np.dot(c_xy - mid, lat_hat))
    if str(diverge_tag).lower() == 'left':
        return d < 0.0
    if str(diverge_tag).lower() == 'right':
        return d > 0.0
    return d >= 0.0


def _mean_min_dist_to_polyline(pts: np.ndarray, line_xy: np.ndarray) -> float:
    """点集到折线的平均最近距离（2D, m）。"""
    if line_xy is None or len(line_xy) < 2 or pts.shape[0] < 1:
        return 1e9
    a = np.asarray(pts, dtype=np.float64)[:, :2]
    b = np.asarray(line_xy, dtype=np.float64)[:, :2]
    line = LineString(b)
    s = 0.0
    for i in range(a.shape[0]):
        s += float(line.distance(Point(float(a[i, 0]), float(a[i, 1]))))
    return s / max(1, a.shape[0])


def pick_second_divider_same_side(
    diverge_tag: str,
    left_boundary_pts,
    right_boundary_pts,
    diverge_boundary_pts,
    gt_fixed_num_sampled_points,
    gt_labels_1d,
    divider_label: int = 0,
):
    """P1 在 left 或 right 的 diverge 路沿上时，在**同一侧**的 GT 车道线(divider)里
    自路沿向路心排序，取**第二条**折线；若 P1=right 则只在 right 半侧找，P1=left 同。

    使用全图 map GT 的 ``label=divider_label``(默认 0) 的折线；以左右路沿均值为「路心」
    近似，用 diverge 侧心点确定半空间，用向路心投影 t 对 divider 从近到远排序。

    Returns:
        (poly_xy, reason) 若失败 (None, reason) 需由调用方回退。
    """
    l_poly = np.asarray(left_boundary_pts, dtype=np.float64)[:, :2]
    r_poly = np.asarray(right_boundary_pts, dtype=np.float64)[:, :2]
    d_poly = np.asarray(diverge_boundary_pts, dtype=np.float64)[:, :2]
    if l_poly.shape[0] < 2 or r_poly.shape[0] < 2 or d_poly.shape[0] < 2:
        return None, '路沿/ diverge 折线过短'
    ninst = len(gt_fixed_num_sampled_points) if gt_fixed_num_sampled_points is not None else 0
    if ninst < 1:
        return None, '无 GT 折线'

    if hasattr(gt_labels_1d, 'cpu'):
        labels = gt_labels_1d.detach().cpu().numpy().reshape(-1)
    else:
        labels = np.asarray(gt_labels_1d, dtype=np.int64).reshape(-1)

    rc = 0.5 * (l_poly.mean(axis=0) + r_poly.mean(axis=0))
    d_center = d_poly.mean(axis=0)
    n_lat = d_center - rc
    nl = float(np.linalg.norm(n_lat))
    if nl < 1e-3:
        return None, '路心估计退化'
    n_lat = n_lat / nl
    n_in = rc - d_center
    ni = float(np.linalg.norm(n_in))
    if ni < 1e-3:
        return None, '向心方向退化'
    n_in = n_in / ni

    rows = []  # (t_in, idx, centroid for debug)
    for i in range(min(ninst, len(labels))):
        if int(labels[i]) != int(divider_label):
            continue
        bb = _tensor_list_to_line_xy(gt_fixed_num_sampled_points, i)
        if bb is None:
            continue
        c = bb.mean(axis=0)
        if float(np.dot(c - rc, n_lat)) < 0.0:
            continue
        t_in = float(np.dot(c - d_center, n_in))
        rows.append((t_in, i, c))
    if not rows:
        return None, f'同侧无可用 divider(Label={divider_label})'

    rows.sort(key=lambda x: x[0])
    if len(rows) < 2:
        return None, f'同侧仅 {len(rows)} 条 divider, 需≥2 才取第2条'

    j = rows[1][1]
    out = _tensor_list_to_line_xy(gt_fixed_num_sampled_points, j)
    if out is None:
        return None, '第2条折线解包失败'
    return out, f'P1 侧 {diverge_tag} 第2条车道线 (div_idx={j})'


def pick_curved_divider_same_side(
    diverge_tag: str,
    left_boundary_pts,
    right_boundary_pts,
    diverge_boundary_pts,
    reference_boundary_pts,
    gt_fixed_num_sampled_points,
    gt_labels_1d,
    allowed_labels: tuple = (0, 1, 2),
    exclude_edge_overlap_m: float = 0.8,
    min_shapely_len_m: float = 1.0,
):
    """P2: 与 P1 **同侧(左/右由 diverge_tag)**，在 GT 矢量线中挑**弯道**最明显的一条（非 P1 路沿）。

    - 同侧用左右路沿中点与横向左→右向量，按 ``dot(c-mid, lat)`` 符号分左/右半幅；
    - 候选为 ``labels in allowed``（nusc: 0/1/2=divider/…/boundary 等，视数据而定）；
    - 排除与 **diverge 路沿** 或 **reference 对侧路沿** 重合的线（点-线平均距离 < 门限，避免 P2 落在 P1 同一条边）；
    - 弯度优先 ``max|k|``，辅以 ``mean|k|``。
    """
    d_poly = np.asarray(diverge_boundary_pts, dtype=np.float64)[:, :2]
    r_poly = np.asarray(reference_boundary_pts, dtype=np.float64)[:, :2]
    if d_poly.shape[0] < 2 or r_poly.shape[0] < 2:
        return None, 'diverge/ reference 路沿过短'
    frame = _road_lateral_frame(left_boundary_pts, right_boundary_pts)
    if frame is None:
        return None, '左右路沿退化'
    mid, lat_hat = frame

    ninst = len(gt_fixed_num_sampled_points) if gt_fixed_num_sampled_points is not None else 0
    if ninst < 1:
        return None, '无 GT 折线'

    if hasattr(gt_labels_1d, 'cpu'):
        labels = gt_labels_1d.detach().cpu().numpy().reshape(-1)
    else:
        labels = np.asarray(gt_labels_1d, dtype=np.int64).reshape(-1)

    allowed = {int(x) for x in allowed_labels}

    rows: list = []  # ( -kmax_tie, -kmax, -kmean, t_in, idx, kmax, kmean, ... )
    for i in range(min(ninst, len(labels))):
        if int(labels[i]) not in allowed:
            continue
        bb = _tensor_list_to_line_xy(gt_fixed_num_sampled_points, i)
        if bb is None or bb.shape[0] < 3:
            continue
        if LineString(bb).length < min_shapely_len_m:
            continue
        c = bb.mean(axis=0)
        if not _on_same_side_as_diverge_tag(c, diverge_tag, mid, lat_hat):
            continue
        # 勿与 P1 所在 diverge 路沿/对侧 reference 路沿 选成同一条
        mdd = _mean_min_dist_to_polyline(bb, d_poly)
        mdr = _mean_min_dist_to_polyline(bb, r_poly)
        if mdd < exclude_edge_overlap_m or mdr < exclude_edge_overlap_m:
            continue
        d_center = d_poly.mean(axis=0)
        rc_ = 0.5 * (
            np.mean(np.asarray(left_boundary_pts, dtype=np.float64)[:, :2], axis=0)
            + np.mean(np.asarray(right_boundary_pts, dtype=np.float64)[:, :2], axis=0)
        )
        n_in = rc_ - d_center
        ni = float(np.linalg.norm(n_in))
        if ni > 1e-6:
            n_in = n_in / ni
            t_in = float(np.dot(c - d_center, n_in))
        else:
            t_in = 0.0
        ks = _curvature_2d(bb)
        k_mean = float(np.mean(np.abs(ks)))
        k_max = float(np.max(np.abs(ks)))
        rows.append((-k_max, -k_mean, t_in, int(i), k_max, k_mean))

    if not rows:
        return None, '同侧无与路沿不重合的可用 GT 线(或全被标签过滤)'

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    j = int(rows[0][3])
    kmax_sel, kmean_sel = float(rows[0][4]), float(rows[0][5])
    out = _tensor_list_to_line_xy(gt_fixed_num_sampled_points, j)
    if out is None:
        return None, '同侧弯道线解包失败'
    return (
        out,
        f'P2 同侧最弯线 bboxes[{j}] lab={int(labels[j])} max|k|={kmax_sel:.4f} mean|k|={kmean_sel:.4f}',
    )


def build_p2_line_stitched_spine_second_edge(
    diverge_boundary_pts,
    reference_boundary_pts,
    step: int = 5,
):
    """asymmetric 时 loss 脊柱(=get_target) 由 **两段** 在一条脊线上拼成: diverge
    前 ``step`` 点 + 平移后的 reference 尾(``y>=y_cut``)。P1 在真 diverge 上搜, P2
    应在这条脊线的 **第二段**（与 diverge 段在接缝处相连之另一边），**不是** 对真
    reference 折线做正交投影的裁切。

    Return:
        (N,2) 的折线, 点数为 0/过短/非 asymmetric 时 ``None``。
    """
    from attack_toolkit.src.utils.utils_attack import get_target_boundary_pts

    d = np.asarray(diverge_boundary_pts, dtype=np.float64)
    r = np.asarray(reference_boundary_pts, dtype=np.float64)
    if d.shape[0] < max(2, step) or r.shape[0] < 2 or d.shape[1] < 2 or r.shape[1] < 2:
        return None
    try:
        full = get_target_boundary_pts(
            d, r, 'left', 'asymmetric', step=step
        )  # tag 在 asymmetric 分支中未用
    except (ValueError, TypeError, IndexError, RuntimeError, KeyError, OSError):
        return None
    arr = np.asarray(full, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < step + 2 or arr.shape[1] < 2:
        return None
    xy = arr[:, :2]
    tail = xy[step:].copy()
    return tail if tail.shape[0] >= 2 else None


def build_p2_search_polyline_2d(
    mode: str | None,
    diverge_boundary_pts,
    reference_boundary_pts,
    diverge_points_list,
    n_diverge_samples: int = 400,
    min_segment_len_m: float = 0.5,
    diverge_tag: str | None = None,
    left_boundary_pts=None,
    right_boundary_pts=None,
    gt_fixed_num_sampled_points=None,
    gt_labels_3d=None,
    loss_spine_xy: np.ndarray | None = None,
    p2_target_far_t0: float = 0.65,
    attack_loss: str | None = None,
    attack_dataset: str | None = None,
    p2_stitch_step: int = 5,
    p2_explicit_gt_bboxes_index: int | None = None,
):
    """P2 在 2D 上沿哪条线密采，与 P1 的「diverge 路沿」并列选项如下。

    - **same_bend**（默认）：:func:`build_opposite_curved_lane_polyline`，对侧 reference
      上与 diverge 弯段横截对齐的**裁切段**（原「对边同弯」）。
    - **full_ref**：**整条** reference 路沿（对侧道沿全长）。
    - **diverge_points**：场景里 ``diverge_points`` 连成的**分/合流折线**（与
      `left`/`right` 路沿并列、描述分流区的另一条几何线，非对侧路沿）。
    - **same_side_2nd_divider**：P1 在 left 则 P2 只在 left 半侧的 GT
      `label=0`（divider）里、自路沿向心排序取**第2条**；P1=right 同理。不足则回退。
    - **same_side_curved_divider**：P2 在 **与 diverge_tag 一致的左/右半幅** 内，
      于 GT 矢量（默认 labels 0/1/2）中选 **弯度最大** 的一条；排除与 diverge / reference
      路沿重合的线；**不再**回退到 P2=diverge（失败则走对边同弯回退）。
    - **diverge_same_edge**：P2 与 P1 用**同一条** diverge 路沿密采；阶段二在候选里优先
      **与 P1 处曲率接近** 的点（见 :func:`build_p2_diverge_same_edge` 与
      ``blind_dual.p2_curvature_match_max``）。
    - **target_far_opposite**：与 loss 同一条脊柱(``loss_spine_xy``) 上
      与 P1 所在真 diverge 在脊线接缝处 **相连** 的 **第二段**。
      **RSA+asymmetric**：与 ``get_target_boundary_pts`` 一致
      (``diverge`` 前 ``p2_stitch_step`` 点 + 平移 ref 尾), P2 在接上的
      **第二段(平移 ref, 非对真 reference 做投影裁切)** 上密采。
      **RSA+symmetric** / **ETA** 时无此两拼边, 回退为
      :func:`build_opposite_curved_lane_polyline`（``loss_spine`` 对真 reference）
      。需 ``loss_spine_xy`` 与 attack_loss/attack_dataset。

    - **gt_bboxes_index**：P1 仍只在 diverge 上搜; P2 在 GT 全图
      ``fixed_num_sampled_points[ p2_explicit_gt_bboxes_index ]`` 上密采（与
      可视化/JSON 中 ``bboxes`` 下标 **一致**）。下标由配置里的 token→i 表给出。
      未给下标或取线失败则回退 ``same_bend``。

    对无效组合自动回退到 ``same_bend``（除 ``diverge_same_edge`` 自洽返回 diverge 折线）。

    Returns:
        (line_xy, description_zh): line_xy shape (N, 2).
    """
    m = (mode or 'same_bend').strip().lower()
    valid_modes = (
        'same_bend', 'full_ref', 'diverge_points', 'same_side_2nd_divider',
        'same_side_curved_divider',
        'diverge_same_edge', 'target_far_opposite', 'gt_bboxes_index',
    )
    if m not in valid_modes:
        m = 'same_bend'

    if m == 'gt_bboxes_index':
        if p2_explicit_gt_bboxes_index is not None:
            gi = int(p2_explicit_gt_bboxes_index)
            line = _tensor_list_to_line_xy(gt_fixed_num_sampled_points, gi)
            if line is not None and len(line) >= 2:
                return (
                    line,
                    f'P2: GT 全图 bboxes[{gi}] (与 P1 成对: Diverge+显式下标, 同 JSON 标号)',
                )
        m = 'same_bend'  # 无下标/折线无效 → 对边同弯

    diverge_same_edge_fail: str | None = None
    if m == 'diverge_same_edge':
        div0 = np.asarray(diverge_boundary_pts, dtype=np.float64)
        if div0.size >= 4 and div0.shape[0] >= 2:
            return (
                np.asarray(div0[:, :2], dtype=np.float64),
                'P2 与 P1 同条 diverge 路沿(双眩光均在同侧该边+曲率近优)',
            )
        diverge_same_edge_fail = 'diverge 路沿点不足(需 N≥2 且二维有效)'

    target_far_fail: str | None = None
    if m == 'target_far_opposite' and loss_spine_xy is not None:
        ls_ = (attack_loss or 'rsa').strip().lower()
        ds_ = (attack_dataset or 'asymmetric').strip().lower()
        sp2: np.ndarray | None = None
        if ls_ == 'rsa' and ds_ == 'asymmetric':
            sp2 = build_p2_line_stitched_spine_second_edge(
                diverge_boundary_pts,
                reference_boundary_pts,
                step=int(p2_stitch_step),
            )
        if sp2 is not None and len(sp2) >= 2:
            return (
                sp2,
                'P2: loss 脊柱(拼接)之第二边, 与 diverge 在脊上相连(非真 reference 投影)',
            )
        # symmetric / eta / 尾段过短: 用脊柱-对真 reference 横截对位
        seg = build_opposite_curved_lane_polyline(
            loss_spine_xy,
            reference_boundary_pts,
            n_diverge_samples=n_diverge_samples,
            min_segment_len_m=min_segment_len_m,
        )
        if seg is not None and len(seg) >= 2:
            if ls_ == 'rsa' and ds_ == 'asymmetric' and (sp2 is None or len(sp2) < 2):
                why = '第二边过短或不可用→对真 ref 对位(回退)'
            elif ls_ == 'eta' or ds_ == 'symmetric':
                why = f'{ls_ or "?"}+{ds_} 无两拼边→对真 ref 对位'
            else:
                why = '对真 ref 对位(回退)'
            return (seg, f'P2: {why}')
        target_far_fail = '脊柱第二边/或 真ref 对位 均失败'
    elif m == 'target_far_opposite':
        target_far_fail = '未提供 loss_spine_xy(与单盲/ETA 的 target 折线一致)'

    ref = np.asarray(reference_boundary_pts, dtype=np.float64)
    if m == 'full_ref' and ref.size >= 4 and ref.shape[0] >= 2:
        line = ref[:, :2] if ref.shape[1] >= 2 else ref
        return np.asarray(line, dtype=np.float64), '对侧路沿 reference 全长'

    if m == 'diverge_points' and diverge_points_list is not None:
        dp = np.asarray(diverge_points_list, dtype=np.float64)
        if dp.ndim == 2 and dp.shape[0] >= 2 and dp.shape[1] >= 2:
            return np.asarray(dp[:, :2], dtype=np.float64), 'diverge_points 分/合流折线'

    same_side_fail: str | None = None
    if m in ('same_side_2nd_divider', 'same_side_curved_divider'):
        if (
            diverge_tag
            and left_boundary_pts is not None
            and right_boundary_pts is not None
            and gt_fixed_num_sampled_points is not None
            and gt_labels_3d is not None
        ):
            if m == 'same_side_curved_divider':
                poly, smsg = pick_curved_divider_same_side(
                    str(diverge_tag),
                    left_boundary_pts,
                    right_boundary_pts,
                    diverge_boundary_pts,
                    reference_boundary_pts,
                    gt_fixed_num_sampled_points,
                    gt_labels_3d,
                )
            else:
                poly, smsg = pick_second_divider_same_side(
                    str(diverge_tag),
                    left_boundary_pts,
                    right_boundary_pts,
                    diverge_boundary_pts,
                    gt_fixed_num_sampled_points,
                    gt_labels_3d,
                )
            if poly is not None and len(poly) >= 2:
                return poly, smsg
            same_side_fail = smsg
        else:
            same_side_fail = '缺少 diverge_tag/左右路沿或 GT 折线(label)'
    out = build_opposite_curved_lane_polyline(
        diverge_boundary_pts,
        reference_boundary_pts,
        n_diverge_samples=n_diverge_samples,
        min_segment_len_m=min_segment_len_m,
    )
    note = '对边同弯 reference 子段'
    if m == 'full_ref':
        note = '对边同弯 (full_ref 不可用，已回退)'
    elif m == 'diverge_points':
        note = '对边同弯 (diverge_points 不足，已回退)'
    elif m == 'same_side_2nd_divider' and same_side_fail is not None:
        note = f'对边同弯 (same_side_2nd_divider 无可用: {same_side_fail})'
    elif m == 'same_side_curved_divider' and same_side_fail is not None:
        note = f'对边同弯 (same_side_curved_divider 无可用: {same_side_fail})'
    elif m == 'diverge_same_edge' and diverge_same_edge_fail is not None:
        note = f'对边同弯 (diverge_same_edge: {diverge_same_edge_fail}，已回退)'
    elif m == 'target_far_opposite' and target_far_fail is not None:
        note = f'对边同弯 (target_far_opposite: {target_far_fail}，已回退)'
    return out, note


def build_p2_on_reference_boundary(
    attack_loc_candidates_on_reference,
    loc1,
    total_locs,
    min_pair_sep_m,
):
    """第二眩光候选：仅在 *对边同弯段* 折线（由 build_opposite_curved_lane_polyline 展宽）上搜索。

    每一项为 (3D, combined_score)。与 P1 距 ≥ min_pair 优先；否则按与 P1 远近来截断。
    """
    assert loc1 is not None

    def far_enough(p):
        return float(np.linalg.norm(p[:2] - loc1[:2])) >= min_pair_sep_m

    cands = sorted(attack_loc_candidates_on_reference, key=lambda x: x[1], reverse=True)

    tier1 = [(p, s) for p, s in cands if far_enough(p)]
    if tier1:
        return [p for p, _ in tier1[:total_locs]]

    with_d = [(p, s, float(np.linalg.norm(p[:2] - loc1[:2]))) for p, s in cands]
    with_d.sort(key=lambda x: x[2], reverse=True)
    if with_d:
        return [p for p, _, _ in with_d[:total_locs]]

    return []


def build_p2_diverge_same_edge(
    attack_loc_candidates,
    loc1,
    total_locs: int,
    min_pair_sep_m: float,
    diverge_boundary_pts_xy: np.ndarray,
    curvature_match_max: float | None = None,
):
    """P2 与 P1 在同一条 diverge 路沿上；优先 |κ(近 P2) - κ(近 P1)| 小，且 2D 距 ≥ min_pair_sep。

    diverge_boundary_pts_xy: (N,2) 与 P1 密采同一条折线（如 right/left 的 diverge 边）。
    curvature_match_max: 若 >0，先只保留曲率差 ≤ 该值；若 None 或 ≤0 则不过滤，仅按曲率差升序截断。

    若曲率/距离无可用候选，回退为 :func:`build_p2_on_reference_boundary` 的排序逻辑（仍用本列表）。
    """
    div = np.asarray(diverge_boundary_pts_xy, dtype=np.float64)
    if div.shape[0] < 2:
        return build_p2_on_reference_boundary(
            attack_loc_candidates, loc1, total_locs, min_pair_sep_m
        )

    k = _curvature_2d(div)
    p1_xy = np.asarray(loc1[:2], dtype=np.float64)

    def far_enough(p) -> bool:
        return float(np.linalg.norm(p[:2] - p1_xy)) >= min_pair_sep_m

    i1 = int(np.argmin(np.sum((div - p1_xy) ** 2, axis=1)))
    i1 = min(max(0, i1), int(k.shape[0]) - 1)
    k1 = float(k[i1])

    def kdiff(p) -> float:
        j = int(np.argmin(np.sum((div - np.asarray(p[:2], dtype=np.float64)) ** 2, axis=1)))
        j = min(max(0, j), int(k.shape[0]) - 1)
        return abs(float(k[j]) - k1)

    cands = sorted(attack_loc_candidates, key=lambda x: x[1], reverse=True)

    scored: list = []
    for p, s in cands:
        if not far_enough(p):
            continue
        scored.append((p, s, kdiff(p)))

    if not scored:
        return build_p2_on_reference_boundary(
            attack_loc_candidates, loc1, total_locs, min_pair_sep_m
        )

    if curvature_match_max is not None and float(curvature_match_max) > 0:
        tier = [t for t in scored if t[2] <= float(curvature_match_max)]
        if tier:
            tier.sort(key=lambda t: t[2])
            return [t[0] for t in tier[: int(total_locs)]]

    scored.sort(key=lambda t: t[2])
    return [t[0] for t in scored[: int(total_locs)]]
