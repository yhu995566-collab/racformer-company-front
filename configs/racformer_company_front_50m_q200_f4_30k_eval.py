import os as _os

_base_ = ['./racformer_company_front_50m_q200_f4.py']

dataset_root = _os.path.abspath(_os.environ.get(
    'RACFORMER_COMPANY_30K_PROCESSED_ROOT',
    '/mnt/diskNvme1/hyh/data/company_20260818_30k_front50_q200_f4/processed'
)) + '/'

point_cloud_range = [0.0, -20.0, -3.0, 50.0, 20.0, 3.0]
grid_config = {
    'x': [0.0, 50.0, 0.5],
    'y': [-20.0, 20.0, 0.5],
    'z': [-3.0, 3.0, 6.0],
    'depth': [1.0, 55.0, 96.0],
    'rcs': [-64, 64, 64],
}
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# The new capture uses the undistorted 1920x1080 front120 camera.  Keep the
# trained network input at 640x256; only the source-image geometry changes.
ida_aug_conf = {
    'resize_lim': (0.334, 0.38),
    'final_dim': (256, 640),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080,
    'W': 1920,
    'rand_flip': True,
}

test_pipeline = [
    dict(type='LoadFrontCameraSweeps', sweeps_num=3),
    dict(type='LoadCompanyRadarSweeps', sweeps_num=3, load_dim=7,
         roi=point_cloud_range),
    dict(type='LoadCompanyLidarPoints', load_dim=5, use_dim=5,
         roi=point_cloud_range),
    dict(type='FrontViewFilter', roi=point_cloud_range),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf,
         training=False),
    dict(type='PointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=1),
    dict(type='RadarPointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=1, test_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RaCFormatBundle3D', class_names=class_names,
                 with_label=False),
            dict(type='Collect3D',
                 keys=['img', 'gt_depth', 'radar_points', 'radar_depth',
                       'radar_rcs'],
                 meta_keys=(
                     'filename', 'box_type_3d', 'ori_shape', 'img_shape',
                     'pad_shape', 'lidar2img', 'img_timestamp',
                     'intrinsics'))
        ])
]

data = dict(
    val=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'custom_infos_val_sweep.pkl',
        pipeline=test_pipeline),
    test=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'custom_infos_test_sweep.pkl',
        pipeline=test_pipeline))

evaluation_output_dir = (
    'outputs/racformer_company_front_50m_q200_f4_30k_eval/evaluation/')

# Delivery evaluation always reports these two scopes from one inference run.
evaluation_profiles = ['car_only', 'main3']
