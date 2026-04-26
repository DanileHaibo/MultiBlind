_base_ = './maptrv2_nusc_r50_24ep.py'

# Use the same 100 asymmetric cases as the current attack artifact while keeping
# the official MapTRv2 model/checkpoint unchanged.
dataset_type = 'CustomNuScenesLocalMapDataset'
asymmetric_dataset_root = 'dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
map_classes = ['divider', 'ped_crossing', 'boundary']

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=map_classes),
            dict(
                type='CustomCollect3D',
                keys=['img'],
                meta_keys=(
                    'filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                    'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'scene_token', 'can_bus',
                    'lidar2global', 'camera2ego', 'camera_intrinsics',
                    'img_aug_matrix', 'lidar2ego', 'global2img',
                )),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        ann_file=asymmetric_dataset_root + 'nuscenes_infos_temporal_val_maptr_asymmetric.pkl',
        map_ann_file=asymmetric_dataset_root + 'nuscenes_map_anns_val_asymmetric.json',
        pipeline=test_pipeline,
        samples_per_gpu=1,
    ),
    test=dict(
        type=dataset_type,
        ann_file=asymmetric_dataset_root + 'nuscenes_infos_temporal_val_maptr_asymmetric.pkl',
        map_ann_file=asymmetric_dataset_root + 'nuscenes_map_anns_val_asymmetric.json',
        pipeline=test_pipeline,
        samples_per_gpu=1,
    ),
)
