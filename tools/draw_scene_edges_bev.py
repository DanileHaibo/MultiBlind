#!/usr/bin/env python3
"""
在 BEV 上绘制「整条地图/矢量线」的**所有**折线并编号，便于你指定 P2 落在第几条边。

数据源（--source）:
  gt   — 默认. 用攻击结果里保存的 `results/map/gt/<token>.json` 的 `bboxes`+`labels`，
         即本帧 GT 中**所有**线要素，通常最完整.
  scene — 仅用 `dataset/scenes_*/<token>.json` 的 left / right + diverge_points
  all  — 先画 GT(编号 #0..)，再画 scene(编号接续)，用线型区分（gt 实线, scene 虚线）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

# 避免「豆腐块」：不要用不存在的字体名；用 fontManager 里实际有的 CJK 字体
def _setup_matplotlib_chinese_font() -> str | None:
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass
    available = {f.name for f in fm.fontManager.ttflist}
    for name in (
        'Noto Sans CJK SC',
        'Noto Sans CJK TC',
        'Noto Sans CJK JP',  # 很多 Linux 上 TTC 只登记为 JP，但含简繁汉字形
        'Noto Sans CJK KR',
        'Noto Serif CJK SC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'AR PL UKai CN',
        'AR PL UMing CN',
    ):
        if name in available:
            matplotlib.rcParams['font.sans-serif'] = [name, 'DejaVu Sans', 'Bitstream Vera Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            return name
    for f in fm.fontManager.ttflist:
        n = f.name or ''
        if 'Noto Sans CJK' in n or 'Noto Serif CJK' in n:
            matplotlib.rcParams['font.sans-serif'] = [n, 'DejaVu Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['axes.unicode_minus'] = False
            return n
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None


_MATPLOTLIB_FONT = _setup_matplotlib_chinese_font()

REPO = Path(__file__).resolve().parent.parent

_EDGE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628',
    '#f781bf', '#999999', '#8dd3c7', '#fb8072', '#80b1d3', '#fdb462',
    '#a6cee3', '#b2df8a', '#fb9a99', '#c9b1ff',
]


def _collect_from_gt(d: dict) -> list[dict]:
    bboxes = d.get('bboxes', [])
    labels = d.get('labels', [])
    out = []
    for i, b in enumerate(bboxes):
        xy = np.asarray(b, dtype=np.float64)
        if xy.size < 4 or xy.shape[0] < 2:
            continue
        lab = int(labels[i]) if i < len(labels) else -1
        out.append(
            {
                'tag': f'gt label={lab}',
                'source': 'gt 全图 bboxes',
                'xy': xy,
                'class_label': lab,
            }
        )
    return out


def _collect_from_scene(scene: dict) -> list[dict]:
    out = []
    for i, el in enumerate(scene.get('map_elements', [])):
        tag = str(el.get('tag', f'el{i}'))
        c = el.get('coordinates', [])
        if not c or len(c) < 2:
            continue
        out.append(
            {
                'tag': tag,
                'source': 'scenes: map_elements',
                'xy': np.asarray(c, dtype=np.float64),
                'class_label': None,
            }
        )
    dpts = scene.get('diverge_points')
    if dpts and len(dpts) >= 2:
        out.append(
            {
                'tag': 'diverge_points',
                'source': 'scenes: diverge_points',
                'xy': np.asarray(dpts, dtype=np.float64),
                'class_label': None,
            }
        )
    return out


def _label_xy(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2:
        return np.mean(xy, axis=0)
    line = LineString(xy)
    p = line.interpolate(0.5 * line.length)
    return np.array([p.x, p.y], dtype=np.float64)


def _load_scene(root: Path, scenes: str, token: str) -> dict | None:
    p = root / 'dataset' / f'scenes_{scenes}' / f'{token}.json'
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def _default_gt_path(root: Path, token: str) -> Path:
    return (
        root
        / 'dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/gt'
        / f'{token}.json'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description='BEV: 全图各边编号 #0 #1 ...')
    ap.add_argument('--root', type=Path, default=REPO)
    ap.add_argument('--token', type=str, required=True)
    ap.add_argument(
        '--source',
        choices=['gt', 'scene', 'all'],
        default='gt',
        help='gt=GT 全 bboxes(默认, 最全); scene=只 scenes JSON; all=GT+scenes 接续编号',
    )
    ap.add_argument(
        '--scenes',
        type=str,
        default='asymmetric',
    )
    ap.add_argument(
        '--gt-json',
        type=Path,
        default=None,
        help='显式指定 gt json；缺省为 train_blind_dual_rsa.../results/map/gt/<token>.json',
    )
    ap.add_argument('-o', '--out', type=Path, default=None)
    ap.add_argument('--best-locs', type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    token = args.token
    edges: list[dict] = []
    sources_note = []

    if args.source in ('gt', 'all'):
        gtp = args.gt_json or _default_gt_path(root, token)
        if not gtp.is_file():
            alts = list(
                (root / 'dataset/maptr-bevpool').glob(f'*/results/map/gt/{token}.json')
            )
            if alts:
                gtp = sorted(alts, key=lambda p: str(p))[-1]
        if gtp.is_file():
            with open(gtp) as f:
                gt = json.load(f)
            e_gt = _collect_from_gt(gt)
            for e in e_gt:
                e['dashed'] = False
                e['id'] = len(edges)
                edges.append(e)
            rel = gtp
            try:
                rel = gtp.relative_to(root)
            except ValueError:
                rel = gtp
            sources_note.append(f'GT: {rel} ({len(e_gt)} 条)')
        else:
            if args.source == 'gt':
                raise SystemExit(
                    f'未找到 GT: {args.gt_json or _default_gt_path(root, token)}\n'
                    f'  可先跑攻击生成 results/map/gt/，或改用 --source scene'
                )

    if args.source in ('scene', 'all') or (args.source == 'gt' and not edges):
        sc = _load_scene(root, args.scenes, token)
        if sc is None:
            if args.source == 'scene' or (args.source == 'gt' and not edges):
                raise SystemExit(
                    f'未找到场景: dataset/scenes_{args.scenes}/{token}.json'
                )
        if sc is not None:
            e_sc = _collect_from_scene(sc)
            n_before = len(edges)
            for e in e_sc:
                # 仅在 all 且前面已有 GT 时，用虚线区分手动 scenes 线
                e['dashed'] = args.source == 'all' and n_before > 0
                e['id'] = len(edges)
                edges.append(e)
            if e_sc:
                sources_note.append(f'scene: {args.scenes} ({len(e_sc)} 条)')

    for e in edges:
        e.setdefault('dashed', False)

    if not edges:
        raise SystemExit('无可用折线，检查 --source / 路径')

    w, h = (14, 12) if len(edges) > 8 else (10, 9)
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    all_pts: list[np.ndarray] = []
    for e in edges:
        xy = e['xy']
        all_pts.append(xy)
        cid = e['id'] % len(_EDGE_COLORS)
        c = _EDGE_COLORS[cid]
        style = dict(lw=1.5 if e.get('dashed') else 2.0, zorder=1, alpha=0.9)
        if e.get('dashed'):
            style['linestyle'] = '--'
        ax.plot(xy[:, 0], xy[:, 1], color=c, **style)
        lab = _label_xy(xy)
        ax.scatter([lab[0]], [lab[1]], c=c, s=160, zorder=3, edgecolor='k', linewidths=0.6)
        ax.annotate(
            f'#{e["id"]}',
            (lab[0], lab[1]),
            xytext=(0, 8),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            color='k',
            ha='center',
            va='bottom',
        )

    blp = args.best_locs
    if blp is None:
        blp = (
            root
            / 'dataset/maptr-bevpool/train_blind_dual_rsa_asymmetric/results/map/attack/best_attack_locs.json'
        )
    if blp.is_file():
        with open(blp) as f:
            bj = json.load(f)
        if token in bj and bj[token][0] is not None:
            a = np.asarray(bj[token][0], dtype=np.float64)
            ax.scatter(
                a[0], a[1], c='#e6007a', s=200, marker='*', zorder=6, edgecolor='w', label='P1',
            )
            if len(bj[token]) > 1 and bj[token][1] is not None:
                b_ = np.asarray(bj[token][1], dtype=np.float64)
                ax.scatter(
                    b_[0], b_[1], c='#00c8ff', s=170, marker='X', zorder=6, edgecolor='k', label='P2',
                )
            ax.legend(loc='lower left', fontsize=8)

    axx = np.concatenate([e[:, 0] for e in all_pts])
    ayy = np.concatenate([e[:, 1] for e in all_pts])
    pad = max(8.0, 0.04 * max(axx.ptp() or 1, ayy.ptp() or 1))
    ax.set_xlim(float(axx.min() - pad), float(axx.max() + pad))
    ax.set_ylim(float(ayy.min() - pad), float(ayy.max() + pad))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    title = f'BEV 全图边 编号  token={token}\n'
    title += f'source={args.source}  |  ' + ' ; '.join(sources_note) if sources_note else ''
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2)

    lines = [f'token: {token}', f'--source {args.source}']
    lines += sources_note
    for e in edges:
        t = f"  #{e['id']}:  {e['tag']!r}  |  {e.get('source','?')}  |  npt={e['xy'].shape[0]}"
        if e.get('class_label') is not None and e['class_label'] >= 0:
            t += f"  (class {e['class_label']})"
        lines.append(t)

    # 不要用 monospace：常见等宽体无 CJK，会导致中文/标签成「豆腐块」
    foot_font = _MATPLOTLIB_FONT or matplotlib.rcParams['font.sans-serif'][0]
    fig.text(
        0.01, 0.01, '\n'.join(lines), fontsize=7, fontfamily=foot_font,
        transform=fig.transFigure, va='bottom', ha='left',
    )
    fig.subplots_adjust(bottom=0.22, left=0.08, right=0.98, top=0.92)

    if args.out is not None:
        out = Path(args.out)
    else:
        out = root / f'scene_full_map_bev_{args.source}_{args.token}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.4)
    print('已写:', out.resolve())
    if _MATPLOTLIB_FONT:
        print('中文字体:', _MATPLOTLIB_FONT)
    print('\n'.join(lines))
    plt.close()


if __name__ == '__main__':
    main()
