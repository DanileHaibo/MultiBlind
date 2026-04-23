#!/usr/bin/env python3
"""
BEV 三列对比：Clean 预测 | Blind(单点) 预测 | Blind-dual(双点) 预测。
在 blind / dual 子图上用与训练一致的攻击位姿（来自 best_attack_locs.json）叠 P1、P2 标记。

需 GPU，与 tools/attack.py 相同的数据与模型配置。示例：

  export PYTHONPATH=$PWD:$PYTHONPATH
  python tools/visualize_clean_blind_triple.py \\
    projects/configs/maptr/maptr_tiny_r50_24e_bevpool_asymmetric.py \\
    ckpts/maptr_tiny_r50_24e_bevpool.pth \\
    --token 0b1c4f8426554e81b723df9e4acf1983 \\
    --blind-locs dataset/maptr-bevpool/train_blind_rsa_asymmetric/results/map/attack/best_attack_locs.json \\
    --dual-locs dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/attack/best_attack_locs.json \\
    -o vis_triple_0b1c.png

  # 或批处理（与 blind/dual 共同 key，默认最多 100 张）:
  # ... 同上 config ckpt，增加: --out-dir vis_triple_100/ --max-samples 100

  # 含 GT 列: 加 --with-gt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet.datasets import replace_ImageToTensor
from PIL import Image

from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from attack_toolkit.src.utils.utils_blind_attack import generate_lens_flare
from attack_toolkit.src.utils.utils_blind_attack_dual import apply_dual_lens_flare
from attack_toolkit.src.utils.utils_attack import plot_bboxes, plot_attack_locs_bev


REPO = Path(__file__).resolve().parents[1]


def _parse_blind_locs_entry(entry) -> np.ndarray | None:
    if entry is None:
        return None
    a = np.asarray(entry, dtype=np.float64).reshape(-1)
    if a.size >= 3:
        return a[:3]
    if a.size == 2:
        return np.append(a, -1.84)
    return None


def _parse_dual_locs_entry(entry) -> tuple[np.ndarray | None, np.ndarray | None]:
    if entry is None or not isinstance(entry, (list, tuple)) or len(entry) < 1:
        return None, None
    p1 = _parse_blind_locs_entry(entry[0]) if entry[0] is not None else None
    p2 = _parse_blind_locs_entry(entry[1]) if len(entry) > 1 and entry[1] is not None else None
    return p1, p2


def _plot_pred_ax(ax, res0, pc_range, car_img, colors_plt, show_score_thr: float) -> None:
    result_dic = res0['pts_bbox']
    sc = result_dic['scores_3d']
    if torch.is_tensor(sc):
        sc = sc.detach().cpu()
    la = result_dic['labels_3d']
    if torch.is_tensor(la):
        la = la.detach().cpu()
    keep = (sc > show_score_thr).numpy()
    pts_3d = result_dic['pts_3d']
    if torch.is_tensor(pts_3d):
        arr_pts = pts_3d.detach().cpu().numpy()
    else:
        arr_pts = np.array(pts_3d)
    arr_lbl = la.numpy() if hasattr(la, 'numpy') else np.asarray(la)
    plot_bboxes(ax, arr_pts[keep], arr_lbl[keep], pc_range, car_img, colors_plt)


def _do_render(
    data,
    model,
    device: torch.device,
    blind_p1,
    dual_p1,
    dual_p2,
    out: Path,
    args,
    pc_range,
    car_img,
    colors_plt,
) -> bool:
    imgs0 = data['img'][0].data[0].clone()
    img_metas = data['img_metas'][0].data[0][0]
    ori_h, ori_w, _ = img_metas['ori_shape'][0]
    img_norm_cfg = img_metas['img_norm_cfg']

    with torch.no_grad():
        data['img'][0].data[0] = imgs0.clone()
        clean = model(return_loss=False, rescale=True, **data)[0]

    with torch.no_grad():
        i1 = imgs0.clone().to(device)
        if blind_p1 is not None:
            light = {
                'position': np.asarray(blind_p1, dtype=np.float64),
                'power': 3000.0,
                'beam_angle': np.radians(40),
            }
            for cam_idx in range(6):
                i1 = generate_lens_flare(
                    i1, img_metas, light, img_norm_cfg, cam_idx, (ori_h, ori_w)
                )
        data['img'][0].data[0] = i1.cpu()
        blind = model(return_loss=False, rescale=True, **data)[0]

    with torch.no_grad():
        i2 = imgs0.clone().to(device)
        if dual_p1 is not None and dual_p2 is not None:
            i2 = apply_dual_lens_flare(
                i2, img_metas, img_norm_cfg, dual_p1, dual_p2, (ori_h, ori_w)
            )
        elif dual_p1 is not None:
            light2 = {
                'position': np.asarray(dual_p1, dtype=np.float64),
                'power': 3000.0,
                'beam_angle': np.radians(40),
            }
            for cam_idx in range(6):
                i2 = generate_lens_flare(
                    i2, img_metas, light2, img_norm_cfg, cam_idx, (ori_h, ori_w)
                )
        data['img'][0].data[0] = i2.cpu()
        dual = model(return_loss=False, rescale=True, **data)[0]

    ncols = 4 if args.with_gt else 3
    fig, axs = plt.subplots(1, ncols, figsize=(3.0 * ncols, 4.0), constrained_layout=True)
    if ncols == 1:
        axs = [axs]
    axl = list(axs)

    cidx = 0
    if args.with_gt:
        gtb = data['gt_bboxes_3d'].data[0][0]
        gtl = data['gt_labels_3d'].data[0][0]
        gt_bboxes = np.array([bbox.cpu().numpy() for bbox in gtb.fixed_num_sampled_points])
        gtl_np = gtl.cpu().numpy() if hasattr(gtl, 'cpu') else np.array(gtl)
        plot_bboxes(axl[0], gt_bboxes, gtl_np, pc_range, car_img, colors_plt)
        axl[0].set_title('GT', fontsize=11)
        cidx = 1

    _plot_pred_ax(
        axl[cidx + 0], clean, pc_range, car_img, colors_plt, args.show_score_thr
    )
    axl[cidx + 0].set_title('Clean', fontsize=11)

    _plot_pred_ax(
        axl[cidx + 1], blind, pc_range, car_img, colors_plt, args.show_score_thr
    )
    axl[cidx + 1].set_title('Blind (1 flare)', fontsize=11)
    if blind_p1 is not None:
        plot_attack_locs_bev(axl[cidx + 1], [blind_p1, None])

    _plot_pred_ax(
        axl[cidx + 2], dual, pc_range, car_img, colors_plt, args.show_score_thr
    )
    axl[cidx + 2].set_title('Blind-dual (2 flares)', fontsize=11)
    if dual_p1 is not None or dual_p2 is not None:
        plot_attack_locs_bev(
            axl[cidx + 2],
            [dual_p1, dual_p2],
        )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print('saved:', out.resolve())
    return True


def _build_data_loader(args, cfg):
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        spg = cfg.data.test.pop('samples_per_gpu', 1)
        if spg > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    return (
        build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        ),
        dataset,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description='BEV: clean vs blind vs blind_dual，叠攻击点 P1/P2'
    )
    ap.add_argument('config', help='maptr test config')
    ap.add_argument('checkpoint', help='.pth')
    ap.add_argument(
        '--token', type=str, default=None, help='单个 sample；与 --out-dir 二选一/并用'
    )
    ap.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='批处理：对 blind/dual 共同 token 各输出 {token}.png，需配合 --max-samples',
    )
    ap.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='批处理时最多条数；默认 100',
    )
    ap.add_argument(
        '--blind-locs',
        type=Path,
        default=REPO
        / 'dataset/maptr-bevpool/train_blind_rsa_asymmetric/results/map/attack/best_attack_locs.json',
    )
    ap.add_argument(
        '--dual-locs',
        type=Path,
        default=REPO
        / 'dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/attack/best_attack_locs.json',
    )
    ap.add_argument(
        '-o', '--out', type=Path, default=None, help='输出 png；默认 vis_triple_{token}.png'
    )
    ap.add_argument('--with-gt', action='store_true', help='增加一列 GT')
    ap.add_argument('--device-id', type=int, default=0)
    ap.add_argument('--show-score-thr', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg['custom_imports'])
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(args.config).split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)

    set_random_seed(args.seed, deterministic=False)
    data_loader, dataset = _build_data_loader(args, cfg)

    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    device = torch.device(f'cuda:{args.device_id}')
    model = MMDataParallel(model, device_ids=[args.device_id])
    model.eval()

    with open(args.blind_locs) as f:
        locs_blind = json.load(f)
    with open(args.dual_locs) as f:
        locs_dual = json.load(f)

    if args.out_dir is None and not args.token:
        raise SystemExit('请指定 --token，或指定 --out-dir 做批处理')
    if args.out_dir is not None and args.token is not None and args.out_dir:
        # 同时指定时：只渲染该 token 到 out-dir
        out_tokens = [args.token]
    elif args.out_dir is not None:
        both = sorted(set(locs_blind) & set(locs_dual))[: int(args.max_samples)]
        out_tokens = both
    else:
        out_tokens = [args.token]
        if out_tokens[0] not in locs_blind or out_tokens[0] not in locs_dual:
            raise KeyError(
                f'token 不在两个 json: blind={out_tokens[0] in locs_blind} dual={out_tokens[0] in locs_dual}'
            )

    token_to_data: dict = {}
    for d in data_loader:
        t = d['img_metas'][0].data[0][0]['sample_idx']
        token_to_data[t] = d

    pc_range = np.array(cfg.point_cloud_range, dtype=np.float64)
    car_path = str(REPO / 'figs' / 'lidar_car.png')
    if not os.path.isfile(car_path):
        car_path = 'figs/lidar_car.png'
    car_img = Image.open(car_path)
    colors_plt = ['orange', 'b', 'g']

    if args.out_dir is not None:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for tkn in out_tokens:
        d_in = token_to_data.get(tkn)
        if d_in is None:
            print(f'[skip] 无 dataloader: {tkn}', file=sys.stderr)
            continue
        blind_p1 = _parse_blind_locs_entry(locs_blind.get(tkn))
        dual_p1, dual_p2 = _parse_dual_locs_entry(locs_dual.get(tkn))
        if args.out_dir is not None:
            outp = Path(args.out_dir) / f'{tkn}.png'
        else:
            outp = args.out or (REPO / f'vis_triple_{tkn}.png')
        if _do_render(
            d_in,
            model,
            device,
            blind_p1,
            dual_p1,
            dual_p2,
            outp,
            args,
            pc_range,
            car_img,
            colors_plt,
        ):
            n_ok += 1
    if args.out_dir is not None and len(out_tokens) > 1:
        print(f'batch: {n_ok}/{len(out_tokens)} → {Path(args.out_dir).resolve()}')


if __name__ == '__main__':
    if str(REPO) not in os.environ.get('PYTHONPATH', ''):
        print(
            f'[hint] 请在仓库根目录执行: export PYTHONPATH={REPO}:$PYTHONPATH',
            file=sys.stderr,
        )
    main()
