# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor

# load attack apis
from mmdetection3d.mmdet3d.apis import single_gpu_attack_patch
from mmdetection3d.mmdet3d.apis import single_gpu_attack_camera_blind
from mmdetection3d.mmdet3d.apis.train_blind_attack_dual import (
    single_gpu_attack_camera_blind_dual,
)
from mmdetection3d.mmdet3d.apis.train_blind_attack_dual_eta import (
    single_gpu_attack_camera_blind_dual_eta,
)

# # debug settings
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("127.0.0.1", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # attack args
    parser.add_argument('--attack_config_file', type=str)
    parser.add_argument(
        '--attack-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the attack config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overridden is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument(
        '--skip-clean-eval',
        action='store_true',
        help='与 --eval 联用：只计算带攻击的 map mAP，跳过 clean 基线（省一次 dataset.evaluate）',
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        
        ### prepare cfg ###
        # read attack_cfg.yaml and add to cfg
        attack_cfg = Config.fromfile(args.attack_config_file)
        cfg.attack = attack_cfg
        cfg.attack_config_file = args.attack_config_file
        
        # Override with any options provided in cfg-options
        if args.attack_options is not None:
            for key, value in args.attack_options.items():
                keys = key.split('.')
                cfg.attack[keys[-1]] = value
        ### prepare cfg ###
        
        model = MMDataParallel(model, device_ids=[args.device_id])
        
        ### attack ###
        if cfg.attack.type == 'patch':
            
            # init output dir
            args.show_dir = os.path.join(args.show_dir, f'train_patch')
            if cfg.attack.loss in ['rsa', 'eta']:
                args.show_dir = args.show_dir + f'_{cfg.attack.loss}'
            else:
                raise ValueError(f'attack loss {cfg.attack.loss} not supported')
            if cfg.attack.dataset != '':
                args.show_dir = args.show_dir + f'_{cfg.attack.dataset}'
            else:
                raise ValueError(f'attack dataset {cfg.attack.dataset} not supported')
            if getattr(cfg.attack, 'tag', None):
                args.show_dir = args.show_dir + f'_{cfg.attack.tag}'
            
            outputs, orig_outputs = single_gpu_attack_patch(model, data_loader, cfg, args.show, args.show_dir)
            
        elif cfg.attack.type == 'blind':
            
            # init output dir
            args.show_dir = os.path.join(args.show_dir, f'train_blind')
            if cfg.attack.loss in ['rsa', 'eta']:
                args.show_dir = args.show_dir + f'_{cfg.attack.loss}'
            else:
                raise ValueError(f'attack loss {cfg.attack.loss} not supported')
            if cfg.attack.dataset != '':
                args.show_dir = args.show_dir + f'_{cfg.attack.dataset}'
            else:
                raise ValueError(f'attack dataset {cfg.attack.dataset} not supported')
            if getattr(cfg.attack, 'tag', None):
                args.show_dir = args.show_dir + f'_{cfg.attack.tag}'
                
            outputs, orig_outputs = single_gpu_attack_camera_blind(model, data_loader, cfg, args.show, args.show_dir)

        elif cfg.attack.type == 'blind_dual':

            # init output dir (same pattern as blind, prefix train_blind_dual)
            args.show_dir = os.path.join(args.show_dir, f'train_blind_dual')
            if cfg.attack.loss in ['rsa', 'eta']:
                args.show_dir = args.show_dir + f'_{cfg.attack.loss}'
            else:
                raise ValueError(f'attack loss {cfg.attack.loss} not supported')
            if cfg.attack.dataset != '':
                args.show_dir = args.show_dir + f'_{cfg.attack.dataset}'
            else:
                raise ValueError(f'attack dataset {cfg.attack.dataset} not supported')
            if getattr(cfg.attack, 'tag', None):
                args.show_dir = args.show_dir + f'_{cfg.attack.tag}'

            if cfg.attack.loss == 'eta':
                outputs, orig_outputs = single_gpu_attack_camera_blind_dual_eta(
                    model, data_loader, cfg, args.show, args.show_dir)
            else:
                outputs, orig_outputs = single_gpu_attack_camera_blind_dual(
                    model, data_loader, cfg, args.show, args.show_dir)

        else:
            
            raise ValueError(f'attack type {cfg.attack.type} not supported')
        ### attack ###
        
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        # set output dir
        kwargs['jsonfile_prefix'] = osp.join(args.show_dir, 'results')
        # kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
        #     '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        # if args.format_only:
        #     dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            
            if not args.skip_clean_eval:
                eval_kwargs['jsonfile_prefix'] = osp.join(
                    args.show_dir, 'results', 'map', 'clean'
                )
                clean_eval_results = dataset.evaluate(orig_outputs, **eval_kwargs)
                clean_maps_path = osp.join(
                    args.show_dir, 'results', 'map', 'clean', 'mAPs.json'
                )
                mmcv.dump(clean_eval_results, clean_maps_path)
            eval_kwargs['jsonfile_prefix'] = osp.join(
                args.show_dir, 'results', 'map', 'attack'
            )
            # print(dataset.evaluate(outputs, **eval_kwargs))
            attack_eval_results = dataset.evaluate(outputs, **eval_kwargs)
            attack_maps_path = osp.join(args.show_dir, 'results', 'map', 'attack', 'mAPs.json')
            mmcv.dump(attack_eval_results, attack_maps_path)
            # print(attack_eval_results)

if __name__ == '__main__':
    main()
