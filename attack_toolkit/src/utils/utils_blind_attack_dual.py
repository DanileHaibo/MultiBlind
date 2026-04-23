"""Helpers for dual blind-spot (two lens flares) attacks.

Kept separate from utils_blind_attack.py so single-blind behavior stays unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import LineString, Point

from attack_toolkit.src.utils.utils_attack import get_asymmetry_anchors
from attack_toolkit.src.utils.utils_blind_attack import generate_lens_flare


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
):
    """P2 在 2D 上沿哪条线密采，与 P1 的「diverge 路沿」并列选项如下。

    - **same_bend**（默认）：:func:`build_opposite_curved_lane_polyline`，对侧 reference
      上与 diverge 弯段横截对齐的**裁切段**（原「对边同弯」）。
    - **full_ref**：**整条** reference 路沿（对侧道沿全长）。
    - **diverge_points**：场景里 ``diverge_points`` 连成的**分/合流折线**（与
      `left`/`right` 路沿并列、描述分流区的另一条几何线，非对侧路沿）。
    - **same_side_2nd_divider**：P1 在 left 则 P2 只在 left 半侧的 GT
      `label=0`（divider）里、自路沿向心排序取**第2条**；P1=right 同理。不足则回退。

    对无效组合自动回退到 ``same_bend``。

    Returns:
        (line_xy, description_zh): line_xy shape (N, 2).
    """
    m = (mode or 'same_bend').strip().lower()
    valid_modes = (
        'same_bend', 'full_ref', 'diverge_points', 'same_side_2nd_divider',
    )
    if m not in valid_modes:
        m = 'same_bend'

    ref = np.asarray(reference_boundary_pts, dtype=np.float64)
    if m == 'full_ref' and ref.size >= 4 and ref.shape[0] >= 2:
        line = ref[:, :2] if ref.shape[1] >= 2 else ref
        return np.asarray(line, dtype=np.float64), '对侧路沿 reference 全长'

    if m == 'diverge_points' and diverge_points_list is not None:
        dp = np.asarray(diverge_points_list, dtype=np.float64)
        if dp.ndim == 2 and dp.shape[0] >= 2 and dp.shape[1] >= 2:
            return np.asarray(dp[:, :2], dtype=np.float64), 'diverge_points 分/合流折线'

    same_side_fail: str | None = None
    if m == 'same_side_2nd_divider':
        if (
            diverge_tag
            and left_boundary_pts is not None
            and right_boundary_pts is not None
            and gt_fixed_num_sampled_points is not None
            and gt_labels_3d is not None
        ):
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
