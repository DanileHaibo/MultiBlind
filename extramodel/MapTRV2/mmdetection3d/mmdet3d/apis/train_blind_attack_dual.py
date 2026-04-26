# Copyright (c) OpenMMLab. All rights reserved.
# Dual blind (RSA only): P1 = diverge 上 RSA；P2 = blind_dual.p2_line（同向高曲度/同侧 divider 等）。
# ETA 双盲为独立实现：mmdet3d.apis.train_blind_attack_dual_eta（P2 仅在 reference 边）。

# Does not modify train_blind_attack.py behavior.
import json
import os
from typing import Optional

import cv2
import mmcv
import numpy as np
import torch
from PIL import Image

from attack_toolkit.src.utils.utils_attack import (
    setup_dirs,
    denormalize_img,
    get_asymmetry_anchors,
    get_target_boundary_pts,
    find_best_matching_boundary,
    sample_boundary_at_interval,
    generate_sampled_points,
    chamfer_distance,
    visualize_attack_results,
)
from attack_toolkit.src.utils.utils_blind_attack_dual import (
    apply_dual_lens_flare,
    build_p2_diverge_same_edge,
    build_p2_on_reference_boundary,
    build_p2_search_polyline_2d,
)
from attack_toolkit.src.utils.utils_blind_attack import (
    calculate_combined_score,
    generate_lens_flare,
)

np.set_printoptions(precision=3, suppress=True)


def _get_blind_dual_cfg(cfg):
    """Optional mmcv.Config subsection blind_dual with defaults."""
    defaults = {
        'min_anchor_sep_m': 3.0,
        'min_pair_sep_m': 1.5,
        'p2_line': 'same_side_2nd_divider',
        # p2_line=gt_bboxes_index 时：P2 在 GT bboxes[下标] 上密采，下标来自
        # p2_gt_token_map_path(JSON: token->int) ；缺 token 用 p2_gt_bboxes_index 整型回退(可省略)
        'p2_gt_token_map_path': None,
        'p2_gt_bboxes_index': None,
        # p2_line=diverge_same_edge 时：仅保留 |κ_P2-κ_P1| ≤ 该值；0 或 None=不按阈值、仅按曲率差升序
        'p2_curvature_match_max': 0.25,
        'p2_target_far_t0': 0.65,  # 兼容项；target_far=脊柱全程对位到对侧 ref，不再用后段 t0
        'p2_stitch_step': 5,  # target_far + asymmetric: 与 get_target 的拼接 step 一致
    }
    if hasattr(cfg.attack, 'blind_dual') and cfg.attack.blind_dual is not None:
        bd = cfg.attack.blind_dual
        u = bd if isinstance(bd, dict) else dict(bd)
        for k in defaults:
            if k in u:
                defaults[k] = u[k]
    return defaults


_P2_TOKEN_MAP_RESOLVED: Optional[str] = None
_P2_TOKEN_MAP: dict = {}


def _reset_p2_token_map_cache() -> None:
    """Tests / 多 run 时如需清缓存可调用。单进程一图即可，路径变则重载。"""
    global _P2_TOKEN_MAP_RESOLVED, _P2_TOKEN_MAP
    _P2_TOKEN_MAP_RESOLVED = None
    _P2_TOKEN_MAP = {}


def _resolve_p2_token_map_path(map_path, cfg) -> Optional[str]:
    if not map_path:
        return None
    map_path = os.path.normpath(os.path.expanduser(str(map_path)))
    if os.path.isfile(map_path):
        return map_path if os.path.isabs(map_path) else os.path.abspath(map_path)
    fn = getattr(cfg, 'filename', None) or getattr(cfg, '_filename', None)
    if fn:
        alt = os.path.join(os.path.dirname(str(fn)), map_path)
        if os.path.isfile(alt):
            return os.path.normpath(alt)
    print(
        f'[warn] p2_gt_token_map_path 非文件(已跳过): {map_path!r}',
        flush=True,
    )
    return None


def _get_p2_token_index_map(bd: dict, cfg) -> dict:
    """token(str) -> int, 自 p2_gt_token_map_path 延迟加载、按路径缓存。"""
    global _P2_TOKEN_MAP_RESOLVED, _P2_TOKEN_MAP
    p = bd.get('p2_gt_token_map_path')
    rp = _resolve_p2_token_map_path(p, cfg)
    if not rp:
        return {}
    if _P2_TOKEN_MAP_RESOLVED == rp and _P2_TOKEN_MAP:
        return _P2_TOKEN_MAP
    with open(rp, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    _P2_TOKEN_MAP = {str(k): int(v) for k, v in raw.items()}
    _P2_TOKEN_MAP_RESOLVED = rp
    print(
        f'[blind_dual] 已加载 p2_gt_token_map: {len(_P2_TOKEN_MAP)} 项 ← {rp}',
        flush=True,
    )
    return _P2_TOKEN_MAP


def single_gpu_attack_camera_blind_dual(
    model,
    data_loader,
    cfg,
    show=False,
    out_dir=None,
    show_score_thr=0.3,
):
    cams_dir = os.path.join(out_dir, 'cams')
    vis_seg_dir = os.path.join(out_dir, 'vis_seg')

    map_results_dir = os.path.join(out_dir, 'results', 'map')
    gt_dir = os.path.join(map_results_dir, 'gt')
    clean_dir = os.path.join(map_results_dir, 'clean')
    attack_dir = os.path.join(map_results_dir, 'attack')

    setup_dirs(
        [out_dir, cams_dir, vis_seg_dir, map_results_dir, gt_dir, clean_dir, attack_dir]
    )

    device = torch.device('cuda:{}'.format(model.device_ids[0]))

    pc_range = cfg.point_cloud_range
    car_img = Image.open('./figs/lidar_car.png')
    colors_plt = ['orange', 'b', 'g']

    results = []
    clean_results = []
    clean_losses = []
    losses = []
    best_attack_locs = {}

    dual_cfg = _get_blind_dual_cfg(cfg)
    if getattr(cfg.attack, 'loss', 'rsa') != 'rsa':
        raise ValueError(
            'single_gpu_attack_camera_blind_dual 仅支持 attack.loss=rsa；'
            'ETA 双盲请用 train_blind_attack_dual_eta.single_gpu_attack_camera_blind_dual_eta'
        )

    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        sample_token = data['img_metas'][0].data[0][0]['sample_idx']

        imgs = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0][0]

        img_h, img_w, img_c = img_metas['img_shape'][0]
        ori_h, ori_w, _ = img_metas['ori_shape'][0]
        img_norm_cfg = img_metas['img_norm_cfg']

        gt_bboxes_3d = data['gt_bboxes_3d'].data[0][0]
        gt_labels_3d = data['gt_labels_3d'].data[0][0]

        dataset_dir = os.path.basename(os.path.dirname(os.path.dirname(out_dir)))
        scene_data_path = f'{dataset_dir}/scenes_{cfg.attack.dataset}/{sample_token}.json'
        with open(scene_data_path, 'r') as f:
            scene_label = json.load(f)

        left_boundary_pts = None
        right_boundary_pts = None
        for boundary in scene_label['map_elements']:
            if boundary['tag'] == 'left':
                left_boundary_pts = np.array(boundary['coordinates'])
            elif boundary['tag'] == 'right':
                right_boundary_pts = np.array(boundary['coordinates'])

        diverge_boundary_tag, _, _, _ = scene_label['diverge_boundary_tag']

        if diverge_boundary_tag == 'left':
            diverge_boundary_pts = left_boundary_pts
            reference_boundary_pts = right_boundary_pts
        else:
            diverge_boundary_pts = right_boundary_pts
            reference_boundary_pts = left_boundary_pts
        diverge_boundary_pts_tensor = torch.tensor(diverge_boundary_pts).to(device)
        reference_boundary_pts_tensor = torch.tensor(reference_boundary_pts).to(device)

        asymmetry_anchors = get_asymmetry_anchors(
            diverge_boundary_pts,
            reference_boundary_pts,
            CURVATURE_DIFF_THRESHOLD=0.1,
            top_k=5,
        )
        asymmetry_anchors = np.hstack(
            [asymmetry_anchors, np.ones((asymmetry_anchors.shape[0], 1)) * -1.84]
        )

        target_boundary_pts = get_target_boundary_pts(
            diverge_boundary_pts,
            reference_boundary_pts,
            diverge_boundary_tag,
            cfg.attack.dataset,
            step=5,
        )
        target_boundary_pts = torch.tensor(target_boundary_pts).to(device)

        with torch.no_grad():
            clean_result = model(return_loss=False, rescale=True, **data)

            pred_pts_3d = clean_result[0]['pts_bbox']['pts_3d'].to(device)
            scores_3d = clean_result[0]['pts_bbox']['scores_3d']
            labels_3d = clean_result[0]['pts_bbox']['labels_3d']
            keep = (scores_3d > show_score_thr) & (labels_3d == 2)
            pred_pts_3d = pred_pts_3d[keep]

            diverge_boundary_pts_pred_clean = find_best_matching_boundary(
                pred_pts_3d,
                diverge_boundary_pts_tensor,
                device=device,
            )

            if diverge_boundary_pts_pred_clean is None:
                clean_loss = torch.tensor(20, dtype=torch.float32, device=device)
            else:
                if diverge_boundary_pts_pred_clean[:, 1].max() > 0:
                    diverge_boundary_pts_pred_clean = diverge_boundary_pts_pred_clean[
                        diverge_boundary_pts_pred_clean[:, 1] > 0
                    ]

                clean_loss = chamfer_distance(
                    diverge_boundary_pts_pred_clean, target_boundary_pts, device=device
                )

            clean_losses.append(clean_loss.detach().item())

        total_locs = cfg.attack.blind.total_locs
        sample_interval = cfg.attack.blind.sample_interval
        locs_height_num = cfg.attack.blind.locs_height_num
        samples_per_loc = cfg.attack.blind.samples_per_loc - 1
        sample_range = cfg.attack.blind.sample_range

        dense_attack_locs = sample_boundary_at_interval(
            diverge_boundary_pts, interval=sample_interval
        )

        if cfg.attack.blind.sample:
            all_attack_locs = dense_attack_locs.copy()
            for attack_loc in dense_attack_locs:
                sampled_points = generate_sampled_points(
                    attack_loc, sample_range, samples_per_loc, mode='random'
                )
                all_attack_locs = np.vstack([all_attack_locs, sampled_points])
        else:
            all_attack_locs = dense_attack_locs

        attack_loc_candidates = []
        for attack_loc in all_attack_locs:
            heights = np.linspace(-1.84, 0, locs_height_num)
            for height in heights:
                attack_loc_3d = np.append(attack_loc, height)
                combined_score = calculate_combined_score(
                    attack_loc_3d,
                    img_metas,
                    asymmetry_anchors,
                    max_beam_angle=np.radians(40),
                )
                attack_loc_candidates.append((attack_loc_3d, combined_score))

        attack_loc_candidates.sort(key=lambda x: x[1], reverse=True)
        top_attack_locs = np.array([p for p, _ in attack_loc_candidates[:total_locs]])

        p2_mode_name = str(dual_cfg.get('p2_line', 'same_bend') or 'same_bend')
        p2_dps = scene_label.get('diverge_points')
        p2_t0 = float(dual_cfg.get('p2_target_far_t0', 0.65) or 0.65)
        sp_np = target_boundary_pts.detach().cpu().numpy()
        loss_spine_2d = sp_np[:, :2] if sp_np.ndim == 2 and sp_np.shape[1] >= 2 else None
        p2_stitch = int(dual_cfg.get('p2_stitch_step', 5) or 5)
        p2_ex_idx = None
        if str(p2_mode_name).strip().lower() == 'gt_bboxes_index':
            tmap = _get_p2_token_index_map(dual_cfg, cfg)
            p2_ex_idx = tmap.get(str(sample_token))
            if p2_ex_idx is None and dual_cfg.get('p2_gt_bboxes_index') is not None:
                p2_ex_idx = int(dual_cfg['p2_gt_bboxes_index'])
        p2_lane_line_2d, p2_line_note = build_p2_search_polyline_2d(
            p2_mode_name,
            diverge_boundary_pts,
            reference_boundary_pts,
            p2_dps,
            n_diverge_samples=400,
            diverge_tag=diverge_boundary_tag,
            left_boundary_pts=left_boundary_pts,
            right_boundary_pts=right_boundary_pts,
            gt_fixed_num_sampled_points=gt_bboxes_3d.fixed_num_sampled_points,
            gt_labels_3d=gt_labels_3d,
            loss_spine_xy=loss_spine_2d,
            p2_target_far_t0=p2_t0,
            attack_loss='rsa',
            attack_dataset=str(getattr(cfg.attack, 'dataset', 'asymmetric')),
            p2_stitch_step=p2_stitch,
            p2_explicit_gt_bboxes_index=p2_ex_idx,
        )
        print(
            f'[P2] 搜索线: {p2_line_note} (p2_line={p2_mode_name!r}, '
            f'vertices={len(p2_lane_line_2d)})',
            flush=True,
        )
        dense_ref_locs = sample_boundary_at_interval(
            p2_lane_line_2d, interval=sample_interval
        )
        if cfg.attack.blind.sample:
            all_ref_attack_locs = dense_ref_locs.copy()
            for attack_loc in dense_ref_locs:
                sampled_points = generate_sampled_points(
                    attack_loc, sample_range, samples_per_loc, mode='random'
                )
                all_ref_attack_locs = np.vstack([all_ref_attack_locs, sampled_points])
        else:
            all_ref_attack_locs = dense_ref_locs

        attack_loc_candidates_opposite = []
        for attack_loc in all_ref_attack_locs:
            heights = np.linspace(-1.84, 0, locs_height_num)
            for height in heights:
                attack_loc_3d = np.append(attack_loc, height)
                combined_score = calculate_combined_score(
                    attack_loc_3d,
                    img_metas,
                    asymmetry_anchors,
                    max_beam_angle=np.radians(40),
                )
                attack_loc_candidates_opposite.append((attack_loc_3d, combined_score))
        attack_loc_candidates_opposite.sort(key=lambda x: x[1], reverse=True)

        best_loss_phase1 = 1e10
        best_attack_loc_single = None

        # ----- Phase 1: identical to single blind RSA — best single flare location -----
        print(
            f'\n[Dual phase 1] Single-flare search (same as blind RSA): '
            f'{len(top_attack_locs)} locations for sample {sample_token}...\n'
        )

        for loc_idx, attack_loc in enumerate(top_attack_locs):
            imgs_adv = imgs.clone().detach().to(device)

            light_source_params = {
                'position': attack_loc,
                'power': 3000.0,
                'beam_angle': np.radians(40),
            }
            for cam_idx in range(6):
                imgs_adv = generate_lens_flare(
                    imgs_adv, img_metas, light_source_params, img_norm_cfg, cam_idx, (ori_h, ori_w)
                )

            data['img'][0].data[0] = imgs_adv.detach().cpu()
            result = model(return_loss=False, rescale=True, **data)

            pred_pts_3d = result[0]['pts_bbox']['pts_3d'].to(device)
            scores_3d = result[0]['pts_bbox']['scores_3d']
            labels_3d = result[0]['pts_bbox']['labels_3d']
            keep = (scores_3d > show_score_thr) & (labels_3d == 2)
            pred_pts_3d = pred_pts_3d[keep]

            diverge_boundary_pts_pred_atk = find_best_matching_boundary(
                pred_pts_3d,
                diverge_boundary_pts_tensor,
                device=device,
            )

            if diverge_boundary_pts_pred_atk is None:
                loss = torch.tensor(20, dtype=torch.float32, device=device)

                progress_msg = (
                    f'\rLoc {loc_idx+1}/{len(top_attack_locs)}: Loss={loss.item():.4f} (No boundary), '
                    f'Best={best_loss_phase1.item():.4f}, Clean={clean_loss.item():.4f}'
                )
                print(progress_msg, end='', flush=True)
                break

            if diverge_boundary_pts_pred_atk[:, 1].max() > 0:
                diverge_boundary_pts_pred_atk = diverge_boundary_pts_pred_atk[
                    diverge_boundary_pts_pred_atk[:, 1] > 0
                ]

            loss = chamfer_distance(
                diverge_boundary_pts_pred_atk, target_boundary_pts, device=device
            )

            if loss < best_loss_phase1:
                best_loss_phase1 = loss
                best_attack_loc_single = attack_loc

            progress_msg = (
                f'\rLoc {loc_idx+1}/{len(top_attack_locs)}: Loss={loss.item():.4f}, '
                f'Best={best_loss_phase1.item():.4f}, Clean={clean_loss.item():.4f}'
            )
            print(progress_msg, end='', flush=True)

        loc1 = best_attack_loc_single
        if loc1 is None:
            loc1 = top_attack_locs[0] if len(top_attack_locs) > 0 else None

        # ----- Phase 2: 固定 P1，在 P2 搜索线（p2_line）上选第二眩光 -----
        best_loss = 1e10
        best_pair = (None, None)

        if loc1 is not None:
            p2_m = str(p2_mode_name or 'same_bend').strip().lower()
            if p2_m == 'diverge_same_edge':
                kraw = dual_cfg.get('p2_curvature_match_max', None)
                if kraw is None:
                    kmax = None
                else:
                    try:
                        kmax = float(kraw)
                    except (TypeError, ValueError):
                        kmax = None
                second_locs = build_p2_diverge_same_edge(
                    attack_loc_candidates_opposite,
                    loc1,
                    total_locs,
                    dual_cfg['min_pair_sep_m'],
                    np.asarray(diverge_boundary_pts, dtype=np.float64)[:, :2],
                    curvature_match_max=kmax,
                )
            else:
                second_locs = build_p2_on_reference_boundary(
                    attack_loc_candidates_opposite,
                    loc1,
                    total_locs,
                    dual_cfg['min_pair_sep_m'],
                )
            if len(second_locs) == 0:
                sep = dual_cfg['min_pair_sep_m']
                for p, _ in attack_loc_candidates_opposite:
                    if np.linalg.norm(p[:2] - loc1[:2]) >= sep:
                        second_locs = [p]
                        break
            if len(second_locs) == 0 and len(attack_loc_candidates_opposite) > 0:
                # 整条 reference 上仍都太近时：取离 P1 最远的点
                farthest = max(
                    attack_loc_candidates_opposite,
                    key=lambda x: np.linalg.norm(x[0][:2] - loc1[:2]),
                )[0]
                second_locs = [farthest]
            if len(second_locs) == 0:
                best_loss = best_loss_phase1
                best_pair = (loc1, None)
            print(
                f'\n\n[Dual phase 2] 固定 P1(单盲最优), 在 P2 搜索线 上试 {len(second_locs)} 个候选: '
                f'{p2_line_note!r} (p2_line={p2_mode_name!r}) ...\n'
            )

            for j, attack_loc_2 in enumerate(second_locs):
                imgs_adv = imgs.clone().detach().to(device)
                imgs_adv = apply_dual_lens_flare(
                    imgs_adv,
                    img_metas,
                    img_norm_cfg,
                    loc1,
                    attack_loc_2,
                    (ori_h, ori_w),
                )

                data['img'][0].data[0] = imgs_adv.detach().cpu()
                result = model(return_loss=False, rescale=True, **data)

                pred_pts_3d = result[0]['pts_bbox']['pts_3d'].to(device)
                scores_3d = result[0]['pts_bbox']['scores_3d']
                labels_3d = result[0]['pts_bbox']['labels_3d']
                keep = (scores_3d > show_score_thr) & (labels_3d == 2)
                pred_pts_3d = pred_pts_3d[keep]

                diverge_boundary_pts_pred_atk = find_best_matching_boundary(
                    pred_pts_3d,
                    diverge_boundary_pts_tensor,
                    device=device,
                )

                if diverge_boundary_pts_pred_atk is None:
                    loss = torch.tensor(20, dtype=torch.float32, device=device)

                    progress_msg = (
                        f'\r2nd {j+1}/{len(second_locs)}: Loss={loss.item():.4f} (No boundary), '
                        f'Best={best_loss.item():.4f}, Clean={clean_loss.item():.4f}'
                    )
                    print(progress_msg, end='', flush=True)
                    break

                if diverge_boundary_pts_pred_atk[:, 1].max() > 0:
                    diverge_boundary_pts_pred_atk = diverge_boundary_pts_pred_atk[
                        diverge_boundary_pts_pred_atk[:, 1] > 0
                    ]

                loss = chamfer_distance(
                    diverge_boundary_pts_pred_atk, target_boundary_pts, device=device
                )

                if loss < best_loss:
                    best_loss = loss
                    best_pair = (loc1, attack_loc_2)

                progress_msg = (
                    f'\r2nd {j+1}/{len(second_locs)}: Loss={loss.item():.4f}, '
                    f'Best={best_loss.item():.4f}, Clean={clean_loss.item():.4f}'
                )
                print(progress_msg, end='', flush=True)
        else:
            best_loss = torch.tensor(1e10, device=device)

        print(f'\n\nSample {sample_token} completed - Best dual loss: {best_loss.item():.4f}')
        losses.append(best_loss.detach().item())

        with torch.no_grad():
            imgs_adv = imgs.clone().detach().to(device)
            if best_pair[0] is not None and best_pair[1] is not None:
                imgs_adv = apply_dual_lens_flare(
                    imgs_adv,
                    img_metas,
                    img_norm_cfg,
                    best_pair[0],
                    best_pair[1],
                    (ori_h, ori_w),
                )
            elif loc1 is not None:
                light_source_params = {
                    'position': loc1,
                    'power': 3000.0,
                    'beam_angle': np.radians(40),
                }
                for cam_idx in range(6):
                    imgs_adv = generate_lens_flare(
                        imgs_adv, img_metas, light_source_params, img_norm_cfg, cam_idx, (ori_h, ori_w)
                    )
            data['img'][0].data[0] = imgs_adv.detach().cpu()
            result = model(return_loss=False, rescale=True, **data)

            combined_adv_imgs = []
            for img_idx in range(6):
                img_adv_save = imgs_adv[0, img_idx].clone()
                img_adv_save = denormalize_img(img_adv_save, img_norm_cfg).permute(1, 2, 0)
                img_adv_save = img_adv_save[:ori_h, :ori_w, :]
                img_adv_save = cv2.cvtColor(
                    img_adv_save.detach().cpu().numpy(), cv2.COLOR_RGB2BGR
                )
                combined_adv_imgs.append(img_adv_save)
            row1 = np.hstack([combined_adv_imgs[2], combined_adv_imgs[0], combined_adv_imgs[1]])
            row2 = np.hstack([combined_adv_imgs[4], combined_adv_imgs[3], combined_adv_imgs[5]])
            combined_adv_imgs = np.vstack([row1, row2])
            adv_imgs_path = os.path.join(cams_dir, f'{sample_token}.png')
            cv2.imwrite(adv_imgs_path, combined_adv_imgs)
            print(f'Save adv images to: {adv_imgs_path}')

        gt_data = (gt_bboxes_3d, gt_labels_3d)
        clean_data = clean_result[0]
        attack_data = result[0]

        attack_locs_bev = None
        if best_pair[0] is not None:
            attack_locs_bev = [np.asarray(best_pair[0], dtype=np.float64).reshape(-1)[:3]]
            attack_locs_bev.append(
                np.asarray(best_pair[1], dtype=np.float64).reshape(-1)[:3]
                if best_pair[1] is not None
                else None
            )

        visualize_attack_results(
            gt_data,
            clean_data,
            attack_data,
            vis_seg_dir,
            f'{sample_token}',
            pc_range,
            car_img,
            colors_plt,
            segment=target_boundary_pts.detach().cpu().numpy(),
            show_score_thr=show_score_thr,
            attack_locs_bev=attack_locs_bev,
        )

        if best_pair[0] is not None:
            entry = [np.asarray(best_pair[0]).tolist()]
            if best_pair[1] is not None:
                entry.append(np.asarray(best_pair[1]).tolist())
            else:
                entry.append(None)
            best_attack_locs[sample_token] = entry
        else:
            best_attack_locs[sample_token] = [None, None]

        results.extend(result)
        clean_results.extend(clean_result)

        gt_bboxes = [bbox.tolist() for bbox in gt_bboxes_3d.fixed_num_sampled_points]
        gt_labels = [label.item() for label in gt_labels_3d]
        gt_data_json = {'bboxes': gt_bboxes, 'labels': gt_labels}
        with open(os.path.join(gt_dir, f'{sample_token}.json'), 'w') as f:
            json.dump(gt_data_json, f)

        batch_size = len(clean_result)
        for _ in range(batch_size):
            prog_bar.update()

    with open(os.path.join(attack_dir, 'best_attack_locs.json'), 'w') as f:
        json.dump(best_attack_locs, f)

    return results, clean_results
