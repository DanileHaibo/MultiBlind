#!/usr/bin/env python3
"""
Offline BEV 可视化：双点 blind 选取逻辑（不跑模型）。

--attack-loss rsa：与 train_blind_attack_dual 一致，P2 见 --p2-line（同向/同侧曲率等）。
--attack-loss eta：与 train_blind_attack_dual_eta 一致，P2 固定为整条 reference 边（对侧边界）。

P1：沿场景标注的 diverge 侧（left/right + diverge_boundary_tag）密采+扰动，与单盲一致。
P2（RSA）：见 --p2-line，**target_far_opposite**=loss 脊柱 与 对侧 reference
   横截对位（成对之另一车道边）上密采，见 --p2-attack-loss。同侧第2条需 GT。
   --p2-gt-edge-id 可强制 bboxes 某条边。洋红为 diverge_points(若有)。

默认在 BEV 上叠画 **results/map/gt 中全部 bboxes**（与 draw_scene 一致），
数字 = **bboxes 数组下标**（与 --p2-gt-edge-id 同索引）。`--no-gt-lane-overlay` 可关。

**P2=固定 GT 边(与 Diverge 成对)**：`--p2-line gt_bboxes_index` + `--p2-gt-token-map`（各 token 对应下标 i），
与训练里 `blind_dual.p2_line: gt_bboxes_index` + `p2_gt_token_map_path` 一致。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle
from shapely.geometry import LineString, Point

from attack_toolkit.src.utils.utils_blind_attack_dual import (
    build_p2_diverge_same_edge,
    build_p2_search_polyline_2d,
)


def _build_opposite_curved_lane_polyline(
    diverge_boundary_pts, reference_boundary_pts, n_diverge_samples=400, min_segment_len_m=0.5
):
    """与 `utils_blind_attack_dual.build_opposite_curved_lane_polyline` 一致，避免 import mmdet 链。"""
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

matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Source Han Sans SC',
    'Droid Sans Fallback',
    'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False

REPO = Path(__file__).resolve().parent.parent


def _sample_boundary_at_interval(boundary_pts, interval=0.5):
    boundary_pts = np.asarray(boundary_pts, dtype=np.float64)
    line = LineString(boundary_pts)
    total_length = line.length
    num_points = max(2, int(total_length / interval) + 1)
    sampled = []
    for i in range(num_points):
        distance = i * interval
        if distance > total_length:
            break
        pt = line.interpolate(distance)
        sampled.append((pt.x, pt.y))
    return np.array(sampled, dtype=np.float64)


def _generate_sampled_points(center, grid_size=1.0, num_points=4, mode='random'):
    center = np.asarray(center, dtype=np.float64)
    if mode == 'random':
        x = np.random.rand(num_points) * grid_size - grid_size / 2 + center[0]
        y = np.random.rand(num_points) * grid_size - grid_size / 2 + center[1]
    else:
        raise NotImplementedError(mode)
    return np.stack([x, y], axis=1)


def _load_boundaries(scene_json: dict):
    left_b = right_b = None
    for boundary in scene_json['map_elements']:
        if boundary['tag'] == 'left':
            left_b = np.array(boundary['coordinates'], dtype=np.float64)
        elif boundary['tag'] == 'right':
            right_b = np.array(boundary['coordinates'], dtype=np.float64)
    tag, *_ = scene_json['diverge_boundary_tag']
    if tag == 'left':
        return left_b, right_b, 'left', left_b, right_b
    return right_b, left_b, 'right', right_b, left_b


def _build_p2_on_reference(
    attack_loc_candidates_on_ref, loc1, total_locs, min_pair_sep_m
):
    """同 utils_blind_attack_dual.build_p2_on_reference_boundary 几何逻辑（离线用伪分）。"""

    def far_enough(p):
        return float(np.linalg.norm(p[:2] - loc1[:2])) >= min_pair_sep_m

    cands = sorted(attack_loc_candidates_on_ref, key=lambda x: x[1], reverse=True)
    tier1 = [(p, s) for p, s in cands if far_enough(p)]
    if tier1:
        return [p for p, _ in tier1[:total_locs]]
    with_d = [
        (p, s, float(np.linalg.norm(p[:2] - loc1[:2]))) for p, s in cands
    ]
    with_d.sort(key=lambda x: x[2], reverse=True)
    if with_d:
        return [p for p, _, _ in with_d[:total_locs]]
    return []


def _pseudo_p1(xy):
    """P1 候选用：略向前方加权的伪分，仅作排序用。"""
    return 0.01 * float(xy[1]) + 1.0 / (1.0 + float(np.hypot(xy[0], xy[1] - 1.0)))


def _pseudo_p2_on_ref(xy):
    """P2(对面线) 候选用：与 P1 的伪分尺度无关，仅分档。"""
    return 0.015 * float(xy[1]) + 0.3 / (1.0 + abs(float(xy[0]) - 0.1))


def _setup_mpl_cjk_font() -> None:
    import matplotlib.font_manager as mfm
    try:
        mfm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass
    available = {f.name for f in mfm.fontManager.ttflist}
    for n in (
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'WenQuanYi Micro Hei',
        'AR PL UMing CN',
    ):
        if n in available:
            matplotlib.rcParams['font.sans-serif'] = [n, 'DejaVu Sans', 'Bitstream Vera Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            return
    for f in mfm.fontManager.ttflist:
        n = f.name or ''
        if 'Noto Sans CJK' in n:
            matplotlib.rcParams['font.sans-serif'] = [n, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return


def _load_loss_spine_bev(
    args,
    diverge_pts: np.ndarray,
    ref_pts: np.ndarray,
    div_tag: str,
) -> np.ndarray | None:
    """与 train 中 RSA get_target / ETA 中心线一致的 2D loss 脊柱(用于 BEV 叠画)."""
    al = str(getattr(args, 'p2_attack_loss', None) or 'rsa')
    if al == 'eta':
        rpc = args.root / 'dataset' / f'diverge_route_centerlines_{args.scenes}' / f'{args.token}.json'
        if not rpc.is_file():
            return None
        with open(rpc) as f:
            j = json.load(f)
        arr = np.asarray(j, dtype=np.float64)
        if arr.size < 4 or arr.shape[0] < 2:
            return None
        return arr[:, :2] if arr.shape[1] >= 2 else None
    try:
        from attack_toolkit.src.utils.utils_attack import get_target_boundary_pts
    except (ImportError, ModuleNotFoundError):
        return None

    ds_ = str(getattr(args, 'p2_attack_dataset', None) or 'asymmetric')
    try:
        tb_ = get_target_boundary_pts(
            np.asarray(diverge_pts, dtype=np.float64),
            np.asarray(ref_pts, dtype=np.float64),
            str(div_tag),
            ds_,
            step=5,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    t = np.asarray(tb_, dtype=np.float64)
    if t.size < 4 or t.shape[0] < 2:
        return None
    return t[:, :2] if t.shape[1] >= 2 else None


def _resolve_map_gt_path(args) -> Path | None:
    """与 --p2-gt-edge-id / same_side_2nd 相同：显式 --gt-json 或默认 train_.../gt/<token>.json 或 maptr 下任一路径。"""
    gtp = getattr(args, 'gt_json', None)
    if gtp is None:
        gtp = (
            args.root
            / 'dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/gt'
            / f'{args.token}.json'
        )
    gtp = Path(gtp)
    if gtp.is_file():
        return gtp
    alts = list(
        (args.root / 'dataset/maptr-bevpool').glob(f'*/results/map/gt/{args.token}.json')
    )
    if alts:
        return sorted(alts, key=lambda p: str(p))[-1]
    return None


def _load_indexed_gt_bboxes(gtp: Path) -> list[tuple[int, np.ndarray]]:
    """bboxes 列表中**原始下标 i** 与 polyline，仅跳过无效几何。"""
    with open(gtp) as f:
        gtj = json.load(f)
    bbs = gtj.get('bboxes', [])
    out: list[tuple[int, np.ndarray]] = []
    for i, bb in enumerate(bbs):
        a = np.asarray(bb, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
            continue
        out.append((i, a[:, :2].copy()))
    return out


def _draw_indexed_gt_bboxes_on_ax(
    ax,
    indexed: list[tuple[int, np.ndarray]],
    zorder: float = 0.3,
) -> None:
    if not indexed:
        return
    cmap = plt.get_cmap('tab20')
    for orig_i, line_xy in indexed:
        c = cmap((orig_i % 20) / 20.0)
        ax.plot(
            line_xy[:, 0],
            line_xy[:, 1],
            color=c,
            ls='-',
            lw=0.9,
            alpha=0.7,
            zorder=zorder,
            solid_capstyle='round',
        )
        m = int(line_xy.shape[0] // 2)
        tx, ty = float(line_xy[m, 0]), float(line_xy[m, 1])
        ax.text(
            tx,
            ty,
            str(orig_i),
            fontsize=6,
            fontweight='bold',
            ha='center',
            va='center',
            zorder=zorder + 0.05,
            color='0.1',
            bbox={
                'boxstyle': 'round,pad=0.12',
                'facecolor': (1.0, 1.0, 1.0, 0.7),
                'edgecolor': c,
                'linewidth': 0.4,
            },
        )


def _resolve_default_gt_json_path(args) -> Path | None:
    gtp = args.gt_json
    if gtp is None:
        gtp = (
            args.root
            / 'dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/gt'
            / f'{args.token}.json'
        )
    gtp = Path(gtp)
    if gtp.is_file():
        return gtp
    alts = list(
        (args.root / 'dataset/maptr-bevpool').glob(f'*/results/map/gt/{args.token}.json')
    )
    if alts:
        return sorted(alts, key=lambda p: str(p))[-1]
    return None


def run_one(args) -> None:
    _setup_mpl_cjk_font()
    scene_path = args.root / 'dataset' / f'scenes_{args.scenes}' / f'{args.token}.json'
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    with open(scene_path) as f:
        scene = json.load(f)
    diverge_pts, ref_pts, div_tag, left_b, right_b = _load_boundaries(scene)
    if diverge_pts is None or ref_pts is None:
        raise ValueError('scene 缺少 left/right 边界')

    dps = scene.get('diverge_points') or []
    div_branch: np.ndarray | None = None
    if len(dps) >= 2:
        div_branch = np.asarray(dps, dtype=np.float64)[:, :2]

    with open(args.attack_cfg) as f:
        atk = yaml.safe_load(f)
    blind = atk['blind']
    dual = atk.get('blind_dual', {'min_pair_sep_m': 1.5})

    sample_interval = float(blind['sample_interval'])
    do_sample = bool(blind.get('sample', True))
    samples_sub = int(blind.get('samples_per_loc', 2)) - 1
    sample_range = float(blind.get('sample_range', 1.0))
    locs_height = int(blind.get('locs_height_num', 4))
    total_locs = int(blind.get('total_locs', 400))
    min_pair = float(dual.get('min_pair_sep_m', 1.5))
    p2t0 = float(dual.get('p2_target_far_t0', 0.65) or 0.65)
    p2_stitch = int(dual.get('p2_stitch_step', 5) or 5)
    spine_args = copy.copy(args)
    spine_args.p2_attack_loss = str(
        getattr(args, 'attack_loss', None)
        or getattr(args, 'p2_attack_loss', None)
        or 'rsa'
    )
    loss_spine_bev = _load_loss_spine_bev(spine_args, diverge_pts, ref_pts, div_tag)

    np.random.seed(0)
    # --- P1: diverge ---
    dense_div = _sample_boundary_at_interval(diverge_pts, interval=sample_interval)
    if do_sample and samples_sub > 0:
        all_2d_div = dense_div.copy()
        for c in dense_div:
            all_2d_div = np.vstack(
                [all_2d_div, _generate_sampled_points(c, sample_range, samples_sub, 'random')]
            )
    else:
        all_2d_div = dense_div

    attack_cand_p1 = []
    for xy in all_2d_div:
        for z in np.linspace(-1.84, 0, locs_height):
            p3 = np.append(xy, z)
            attack_cand_p1.append((p3, _pseudo_p1(xy)))
    attack_cand_p1.sort(key=lambda x: x[1], reverse=True)

    # --- P2: RSA 用 blind_dual.p2_line；ETA 双盲固定为 reference 全边（与 train_blind_attack_dual_eta 一致）---
    attack_loss_viz = str(getattr(args, 'attack_loss', 'rsa') or 'rsa').lower()
    bd0 = atk.get('blind_dual') or {}
    ei_tar = None
    ei_eff = None
    p2_ei = None
    p2_ei_from_map = False

    if attack_loss_viz == 'eta':
        p2_lane_2d = np.asarray(ref_pts, dtype=np.float64)[:, :2]
        p2_mode = 'ETA: P2=reference 路沿（对侧边界，与 RSA 双盲 P2 策略解耦）'
        p2_line_key = 'eta_reference'
    else:
        p2_mode_arg = getattr(args, 'p2_line', None)
        if p2_mode_arg is None and isinstance(bd0, dict) and 'p2_line' in bd0:
            p2_mode_arg = str(bd0['p2_line'])
        else:
            p2_mode_arg = str(p2_mode_arg or 'same_bend')
        p2m_low = p2_mode_arg.strip().lower()

        if args.p2_gt_edge_id is not None:
            p2_ei = int(args.p2_gt_edge_id)
        elif p2m_low == 'gt_bboxes_index':
            tm = getattr(args, 'p2_gt_token_map', None)
            if tm and Path(tm).is_file():
                with open(tm, encoding='utf-8') as f:
                    mp = json.load(f)
                if args.token in mp:
                    p2_ei = int(mp[args.token])
                    p2_ei_from_map = True
            if p2_ei is None and getattr(args, 'p2_gt_bboxes_index_fallback', None) is not None:
                p2_ei = int(args.p2_gt_bboxes_index_fallback)
            if p2_ei is None:
                raise SystemExit(
                    'p2_line=gt_bboxes_index: 需在 --p2-gt-token-map 中给当前 token, '
                    '或设 --p2-gt-bboxes-fallback / --p2-gt-edge-id'
                )

        p2_mode = '对边同弯 reference 子段'
        p2_line_key = None
        if p2_ei is not None:
            gtp = _resolve_default_gt_json_path(args)
            if gtp is None or not gtp.is_file():
                raise SystemExit('找不到 GT json（见 --gt-json 或 maptr.../gt/<token>.json）')
            with open(gtp) as f:
                gtj = json.load(f)
            bbs = gtj.get('bboxes', [])
            ei_tar = int(p2_ei)
            nbb = len(bbs)
            if nbb < 1:
                raise SystemExit(f'无 bboxes: {gtp}')
            strict = bool(getattr(args, 'p2_strict', False))
            if not (0 <= ei_tar < nbb):
                if strict:
                    raise SystemExit(
                        f'P2 下标={ei_tar} 越界: GT 共 {nbb} 条: {gtp}'
                    )
                ei_eff = nbb - 1
            else:
                ei_eff = ei_tar
            p2_lane_2d = np.asarray(bbs[ei_eff], dtype=np.float64)[:, :2]
            p2_mode = f'GT 全图边 #{ei_eff}' + (
                f' (n={nbb}条, 目標#{ei_tar} 不可用→末条)' if ei_eff != ei_tar else ''
            )
            if p2_ei_from_map:
                p2_mode += ' [token 映射表]'
            print('P2 GT 文件:', gtp.resolve(), f'| 下标 有效#{ei_eff} 目標#{ei_tar}')
            p2_line_key = None
        else:
            gt_fixed = None
            gt_lbl = None
            if p2_mode_arg.strip().lower() in ('same_side_2nd_divider', 'same_side_curved_divider'):
                gtp2 = _resolve_default_gt_json_path(args)
                if gtp2 is not None and gtp2.is_file():
                    with open(gtp2) as f:
                        gtj = json.load(f)
                    bbs = gtj.get('bboxes', [])
                    labels = gtj.get('labels', [])
                    if bbs and len(bbs) == len(labels):
                        gt_fixed = bbs
                        gt_lbl = np.asarray(labels, dtype=np.int64)
            p2_lane_2d, p2_mode = build_p2_search_polyline_2d(
                p2_mode_arg,
                diverge_pts,
                ref_pts,
                dps,
                n_diverge_samples=400,
                min_segment_len_m=0.5,
                diverge_tag=div_tag,
                left_boundary_pts=left_b,
                right_boundary_pts=right_b,
                gt_fixed_num_sampled_points=gt_fixed,
                gt_labels_3d=gt_lbl,
                loss_spine_xy=loss_spine_bev,
                p2_target_far_t0=p2t0,
                attack_loss=str(getattr(args, 'p2_attack_loss', None) or 'rsa'),
                attack_dataset=str(getattr(args, 'p2_attack_dataset', None) or 'asymmetric'),
                p2_stitch_step=p2_stitch,
            )
            p2_line_key = str(p2_mode_arg or 'same_bend').strip().lower()
    p2_legend = f'P2: {p2_mode}'
    if ei_eff is not None:
        p2_legend = f'P2: GT 边 #{ei_eff}'
        if ei_tar is not None and ei_eff != ei_tar:
            p2_legend += f' (目標#{ei_tar})'
    dense_ref = _sample_boundary_at_interval(p2_lane_2d, interval=sample_interval)
    if do_sample and samples_sub > 0:
        all_2d_ref = dense_ref.copy()
        for c in dense_ref:
            all_2d_ref = np.vstack(
                [all_2d_ref, _generate_sampled_points(c, sample_range, samples_sub, 'random')]
            )
    else:
        all_2d_ref = dense_ref

    attack_cand_p2 = []
    for xy in all_2d_ref:
        for z in np.linspace(-1.84, 0, locs_height):
            p3 = np.append(xy, z)
            attack_cand_p2.append((p3, _pseudo_p2_on_ref(xy)))
    attack_cand_p2.sort(key=lambda x: x[1], reverse=True)

    if getattr(args, 'pseudo_pair', False):
        loc1 = np.array(attack_cand_p1[0][0], dtype=np.float64)
        loc2 = None
    else:
        best_p = args.best_locs
        if best_p is None:
            best_p = (
                args.root
                / 'dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/attack/best_attack_locs.json'
            )
        with open(best_p) as f:
            best_map = json.load(f)
        if args.token not in best_map:
            raise KeyError(f'JSON 中无 {args.token}')
        loc1 = np.array(best_map[args.token][0], dtype=np.float64)
        loc2 = (
            None
            if best_map[args.token][1] is None
            else np.array(best_map[args.token][1], dtype=np.float64)
        )

    kmax_vis = None
    if isinstance(dual, dict) and dual.get('p2_curvature_match_max', None) is not None:
        try:
            kmax_vis = float(dual['p2_curvature_match_max'])
        except (TypeError, ValueError):
            kmax_vis = None

    if p2_line_key == 'diverge_same_edge' and p2_ei is None:
        div_xy = np.asarray(diverge_pts, dtype=np.float64)[:, :2]
        second_locs = build_p2_diverge_same_edge(
            attack_cand_p2,
            loc1,
            total_locs,
            min_pair,
            div_xy,
            curvature_match_max=kmax_vis,
        )
    else:
        second_locs = _build_p2_on_reference(attack_cand_p2, loc1, total_locs, min_pair)
    if len(second_locs) == 0:
        for p, _ in attack_cand_p2:
            if float(np.linalg.norm(p[:2] - loc1[:2])) >= min_pair:
                second_locs = [p]
                break
    if len(second_locs) == 0 and len(attack_cand_p2) > 0:
        second_locs = [
            max(attack_cand_p2, key=lambda x: float(np.linalg.norm(x[0][:2] - loc1[:2])))[0]
        ]

    if getattr(args, 'pseudo_pair', False) and loc2 is None and len(second_locs) > 0:
        loc2 = np.array(second_locs[0], dtype=np.float64)

    second_xy = (
        np.array([p[:2] for p in second_locs]) if len(second_locs) else np.zeros((0, 2))
    )
    # GT 全图 bboxes（下标=编号），与 --p2-gt-edge-id 同索引
    gt_indexed: list[tuple[int, np.ndarray]] = []
    if not getattr(args, 'no_gt_lane_overlay', False):
        _gp = _resolve_map_gt_path(args)
        if _gp is not None and _gp.is_file():
            gt_indexed = _load_indexed_gt_bboxes(_gp)
            print(f'GT 全图 bboxes: {len(gt_indexed)} 条 → {_gp.resolve()}')

    p2_line_for_dist = LineString(p2_lane_2d)
    d_p2_ref = None
    if loc2 is not None:
        d_p2_ref = float(
            p2_line_for_dist.distance(Point(float(loc2[0]), float(loc2[1])))
        )

    # ----- 出图 2x2 -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    span = max(
        max(diverge_pts[:, 0].ptp(), ref_pts[:, 0].ptp()) + 4,
        max(diverge_pts[:, 1].ptp(), ref_pts[:, 1].ptp()) + 4,
        25.0,
    )
    all_xy = np.vstack([diverge_pts, ref_pts])
    if div_branch is not None:
        all_xy = np.vstack([all_xy, div_branch])
    if loss_spine_bev is not None and len(loss_spine_bev) > 0:
        all_xy = np.vstack([all_xy, loss_spine_bev])
    if p2_lane_2d is not None and len(p2_lane_2d) > 0:
        all_xy = np.vstack([all_xy, np.asarray(p2_lane_2d, dtype=np.float64)[:, :2]])
    for _i, gxy in gt_indexed:
        if len(gxy) >= 1:
            all_xy = np.vstack([all_xy, gxy])
    cx, cy = float(np.median(all_xy[:, 0])), float(np.median(all_xy[:, 1]))
    xlim = (cx - span / 2, cx + span / 2)
    ylim = (cy - span / 2, cy + span / 2)

    for ax in axes.ravel():
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    for ax in axes.ravel():
        _draw_indexed_gt_bboxes_on_ax(ax, gt_indexed, zorder=0.3)

    def _plot_div_j(ax, with_label: bool) -> None:
        if div_branch is None or len(div_branch) < 2:
            return
        ax.plot(
            div_branch[:, 0], div_branch[:, 1], color='#aa00cc', ls='-.', lw=1.25, alpha=0.95,
            zorder=2, label='diverge_points(分/合流,非路沿)' if with_label else '_',
        )

    def _draw_loss_spine(ax, label_full: bool) -> None:
        if loss_spine_bev is None or len(loss_spine_bev) < 2:
            return
        ax.plot(
            loss_spine_bev[:, 0],
            loss_spine_bev[:, 1],
            color='#c45c00',
            ls='-',
            lw=1.65,
            alpha=0.9,
            zorder=4.0,
            label='loss 脊柱(与单盲/训练一致)' if label_full else '_',
        )

    ax = axes[0, 0]
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'k--', alpha=0.5, label='Reference（对面道）')
    ax.plot(diverge_pts[:, 0], diverge_pts[:, 1], 'b-', lw=2, label='Diverge（P1 候选折线）')
    _plot_div_j(ax, True)
    _draw_loss_spine(ax, label_full=True)
    ax.scatter(all_2d_div[:, 0], all_2d_div[:, 1], s=2, c='0.45', alpha=0.35, label='P1 2D 候选(密采+扰动)', zorder=1)
    ax.set_title('(1) 阶段一：只沿 diverge 生成 P1 候选，RSA 同单点 blind')
    ax.legend(loc='upper right', fontsize=7)

    ax = axes[0, 1]
    ax.plot(diverge_pts[:, 0], diverge_pts[:, 1], 'b-', alpha=0.3, linewidth=1, label='Diverge（P1）')
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'k:', alpha=0.35, lw=1, label='整条 reference(示意)')
    _plot_div_j(ax, False)
    _draw_loss_spine(ax, label_full=True)
    ax.plot(
        p2_lane_2d[:, 0], p2_lane_2d[:, 1], 'g-', lw=2.4, label=p2_legend,
    )
    ax.scatter(
        all_2d_ref[:, 0], all_2d_ref[:, 1], s=2, c='darkgreen', alpha=0.45, label='P2 密采(仅上列绿线)',
    )
    ax.set_title(
        f'(2) 阶段二：P2 搜索线 = {p2_mode}，再密采+扰动'
    )
    ax.legend(loc='upper right', fontsize=6)

    ax = axes[1, 0]
    ax.plot(diverge_pts[:, 0], diverge_pts[:, 1], 'b-', lw=1.0, alpha=0.4, label='Diverge')
    _plot_div_j(ax, False)
    _draw_loss_spine(ax, label_full=False)
    ax.plot(p2_lane_2d[:, 0], p2_lane_2d[:, 1], 'g-', lw=1.5, alpha=0.8, label='P2 密采线')
    if len(second_xy):
        ax.scatter(
            second_xy[:, 0],
            second_xy[:, 1],
            s=3,
            c='orange',
            alpha=0.45,
            label='阶段二候选(对边同弯子段+伪分截断)',
        )
    c = Circle(
        (float(loc1[0]), float(loc1[1])),
        min_pair,
        color='0.2',
        fill=False,
        ls=':',
        lw=1,
        label=f'与 P1 距<{min_pair}m 禁选',
    )
    ax.add_patch(c)
    ax.scatter(
        [loc1[0]], [loc1[1]], c='#e6007a', s=200, marker='*', zorder=12, label='P1'
    )
    if loc2 is not None:
        ax.scatter(
            [loc2[0]], [loc2[1]], c='#00c8ff', s=200, marker='X', zorder=12, label='P2(JSON)'
        )
    st = (
        f'(3) 在 P2 搜索线(绿) 的候选(橙) 中、与 P1 距≥{min_pair}m 上选 P2；'
    )
    if d_p2_ref is not None and d_p2_ref > 2.0:
        st += f'\n注: JSON 的 P2 至绿线线距约 {d_p2_ref:.1f}m，若为旧 run 则可能不贴线。'
    ax.set_title(st)
    ax.legend(loc='upper right', fontsize=6)

    ax = axes[1, 1]
    ax.plot(diverge_pts[:, 0], diverge_pts[:, 1], 'b-', lw=2, label='Diverge')
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], 'k:', alpha=0.3, label='全 reference(灰)')
    _plot_div_j(ax, False)
    _draw_loss_spine(ax, label_full=False)
    ax.plot(p2_lane_2d[:, 0], p2_lane_2d[:, 1], 'g-', lw=2, label='P2 密采线')
    ax.scatter([loc1[0]], [loc1[1]], c='#e6007a', s=300, marker='*', zorder=12, edgecolor='w', label='P1')
    if loc2 is not None:
        ax.scatter(
            [loc2[0]], [loc2[1]], c='#00c8ff', s=200, marker='X', zorder=12, edgecolor='w', label='P2'
        )
        ax.plot(
            [loc1[0], loc2[0]], [loc1[1], loc2[1]], c='0.3', ls='--', alpha=0.5, zorder=5, label='连线'
        )
    ax.set_title(
        f'(4) 汇总\ntoken={args.token}；P1: diverge；P2: {p2_mode}\n'
        f'真跑时 combined_score 与伪分不同。'
    )
    ax.legend(loc='upper right', fontsize=6)

    out = args.out
    if out is None:
        out = args.root / f'dual_point_selection_viz_{args.token}.png'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _gt_leg = f'  |  彩线+数字=GT 全 bboxes(下标 i)'
    if not gt_indexed:
        _gt_leg = ''
    _mode_leg = 'ETA' if attack_loss_viz == 'eta' else 'RSA'
    fig.suptitle(
        f'双点 blind ({_mode_leg}): P1=diverge；P2={p2_mode}  |  '
        f'土橙=loss 脊柱  绿=P2 密采线{_gt_leg}',
        fontsize=12,
        fontweight='bold',
    )
    fig.savefig(out, dpi=200)
    print(f'已保存: {out.resolve()}')
    print('P2 模式:', p2_mode)
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description='双点 blind P1/P2 离线图（可选 GT 全图边作为 P2 线）',
    )
    ap.add_argument('--root', type=Path, default=REPO, help='仓库根目录')
    ap.add_argument(
        '--token', type=str, default=None, help='sample token（与 nusc 一致，批处理时不必填）'
    )
    ap.add_argument(
        '--scenes',
        type=str,
        default='asymmetric',
        help='与 cfg.attack.dataset 一致，如 asymmetric',
    )
    ap.add_argument('--best-locs', type=Path, default=None, help='best_attack_locs.json')
    ap.add_argument(
        '--attack-cfg',
        type=Path,
        default=REPO / 'attack_toolkit' / 'configs' / 'attack_cfg.yaml',
    )
    ap.add_argument('-o', '--out', type=Path, default=None, help='输出 png 路径')
    ap.add_argument(
        '--p2-line',
        type=str,
        default=None,
        choices=[
            'same_bend',
            'full_ref',
            'diverge_points',
            'same_side_2nd_divider',
            'same_side_curved_divider',
            'diverge_same_edge',
            'target_far_opposite',
            'gt_bboxes_index',
        ],
        help='P2 密采折线；same_side_curved_divider=同侧且优先曲率大。gt_bboxes_index=P2 在 bboxes[下标] 上',
    )
    ap.add_argument(
        '--p2-attack-loss',
        type=str,
        default='rsa',
        choices=['rsa', 'eta'],
        help='p2_line=target_far 时: rsa=get_target 拼接(与单盲一致)；eta=diverge_route_centerlines json',
    )
    ap.add_argument(
        '--p2-attack-dataset',
        type=str,
        default='asymmetric',
        help='--p2-attack-loss=rsa 时 get_target_boundary 的 dataset 名',
    )
    ap.add_argument(
        '--p2-gt-edge-id',
        type=int,
        default=None,
        help='P2 在 GT 全图 bboxes[该下标] 上密采(与图例数字同索引)。指定时优先, 不读 p2-gt-token-map',
    )
    ap.add_argument(
        '--p2-gt-token-map',
        type=Path,
        default=None,
        help='JSON: { "token": 下标int }；配合 --p2-line gt_bboxes_index，P1 仍在 diverge',
    )
    ap.add_argument(
        '--p2-gt-bboxes-fallback',
        type=int,
        default=None,
        help='p2-gt-token-map 无本 token 时的 bboxes 下标(可省略, 无则退出)',
    )
    ap.add_argument(
        '--gt-json',
        type=Path,
        default=None,
        help='显式 GT json；缺省为 train_blind_dual_rsa.../results/map/gt/<token>.json',
    )
    ap.add_argument(
        '--no-gt-lane-overlay',
        action='store_true',
        help='不叠画 GT 全图 bboxes 及编号(默认: 与 draw_scene 一致, 数字=bboxes 下标)',
    )
    ap.add_argument(
        '--p2-strict',
        action='store_true',
        help='P2 目标下标越界时直接失败，不降级为末条',
    )
    ap.add_argument(
        '--batch-out-dir',
        type=Path,
        default=None,
        help='批处理：将每张图写到 <dir>/<token>.png（按 batch-gt-dir 下 token 数，最多 --max-samples 个）',
    )
    ap.add_argument(
        '--batch-gt-dir',
        type=Path,
        default=None,
        help='批处理：从该目录枚举 *.json 的 stem 作为 token；缺省为 train_blind.../results/map/gt',
    )
    ap.add_argument(
        '--max-samples', type=int, default=100, help='批处理时最多张数，默认 100'
    )
    ap.add_argument(
        '--attack-loss',
        type=str,
        default='rsa',
        choices=['rsa', 'eta'],
        help='与 train 一致：rsa=RSA 双盲 P2 策略(见 blind_dual.p2_line)；'
        'eta=ETA 双盲 P2 固定为 reference 边（train_blind_attack_dual_eta）',
    )
    ap.add_argument(
        '--pseudo-pair',
        action='store_true',
        help='不用 best_attack_locs.json：P1=伪分最高候选，P2=在 P2 线上与 P1 满足间距的首个候选',
    )
    return ap


def main() -> None:
    p = _build_parser()
    args = p.parse_args()
    if args.batch_out_dir is not None:
        matplotlib.use('Agg')
        gtd = args.batch_gt_dir or (
            args.root
            / 'dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/gt'
        )
        gtd = Path(gtd)
        toks = sorted(x.stem for x in gtd.glob('*.json'))[: int(args.max_samples)]
        outd = Path(args.batch_out_dir)
        outd.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        fails: list[tuple[str, Exception]] = []
        for t in toks:
            a = copy.copy(args)
            a.token = t
            a.out = outd / f'{t}.png'
            a.batch_out_dir = None
            a.gt_json = None
            try:
                run_one(a)
                n_ok += 1
            except Exception as e:
                fails.append((t, e))
        print(
            f'批处理完成: 成功 {n_ok}/{len(toks)} → {outd.resolve()}'
        )
        if fails:
            print(f'失败 {len(fails)} 个（仅列前 20 个）:')
            for t, e in fails[:20]:
                print(f'  {t}  {e!r}')
        return
    if not args.token:
        p.error('需指定 --token 或使用 --batch-out-dir')
    run_one(args)


if __name__ == '__main__':
    main()
