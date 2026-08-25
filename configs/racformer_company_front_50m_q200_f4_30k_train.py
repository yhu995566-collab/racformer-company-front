import os as _os

_base_ = ['./racformer_company_front_50m_q200_f4.py']

num_frames = 4
num_cams = 1
point_cloud_range = [0.0, -20.0, -3.0, 50.0, 20.0, 3.0]
grid_config = {
    'x': [0.0, 50.0, 0.5],
    'y': [-20.0, 20.0, 0.5],
    'z': [-3.0, 3.0, 6.0],
    'depth': [1.0, 55.0, 96.0],
    'rcs': [-64, 64, 64],
}
evaluation_distance_ranges = [(0, 25), (25, 50), (0, 50)]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

dataset_root = _os.path.abspath(_os.environ.get(
    'RACFORMER_COMPANY_TRAINVAL_ROOT',
    '/mnt/diskNvme1/hyh/data/'
    'company_20260818_30k_front50_q200_f4/processed_trainval_v1')) + '/'
test_root = _os.path.abspath(_os.environ.get(
    'RACFORMER_COMPANY_TEST_ROOT',
    '/mnt/diskNvme1/hyh/data/'
    'company_20260818_30k_front50_q200_f4/processed_v3')) + '/'

# The 30k collection stores native 1920x1080 undistorted images, unlike the
# earlier 640x480 company training set used by the base delivery config.
ida_aug_conf = {
    'resize_lim': (0.334, 0.38),
    'final_dim': (256, 640),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080,
    'W': 1920,
    'rand_flip': True,
}

train_pipeline = [
    dict(type='LoadFrontCameraSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False, with_label=False, with_bbox_depth=False),
    dict(type='LoadCompanyRadarSweeps', sweeps_num=num_frames - 1,
         load_dim=7, roi=point_cloud_range),
    dict(type='LoadCompanyLidarPoints', load_dim=5, use_dim=5,
         roi=point_cloud_range),
    dict(type='FrontViewFilter', roi=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='PointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=num_cams),
    dict(type='RadarPointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=num_cams, test_mode=False),
    dict(type='RaCFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_depth',
               'radar_depth', 'radar_rcs', 'radar_points'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'lidar2img', 'img_timestamp', 'intrinsics')),
]

test_pipeline = [
    dict(type='LoadFrontCameraSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadCompanyRadarSweeps', sweeps_num=num_frames - 1,
         load_dim=7, roi=point_cloud_range),
    dict(type='LoadCompanyLidarPoints', load_dim=5, use_dim=5,
         roi=point_cloud_range),
    dict(type='FrontViewFilter', roi=point_cloud_range),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='PointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=num_cams),
    dict(type='RadarPointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=num_cams, test_mode=True),
    dict(type='MultiScaleFlipAug3D', img_scale=(1920, 1080),
         pts_scale_ratio=1, flip=False, transforms=[
             dict(type='RaCFormatBundle3D', class_names=class_names,
                  with_label=False),
             dict(type='Collect3D',
                  keys=['img', 'gt_depth', 'radar_points', 'radar_depth',
                        'radar_rcs'],
                  meta_keys=('filename', 'box_type_3d', 'ori_shape',
                             'img_shape', 'pad_shape', 'lidar2img',
                             'img_timestamp', 'intrinsics')),
         ]),
]

data = dict(
    train=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'custom_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        num_sweeps=num_frames - 1,
        point_cloud_range=point_cloud_range,
        evaluation_distance_ranges=evaluation_distance_ranges),
    val=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'custom_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        num_sweeps=num_frames - 1,
        point_cloud_range=point_cloud_range,
        evaluation_distance_ranges=evaluation_distance_ranges),
    test=dict(
        data_root=test_root,
        ann_file=test_root + 'custom_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        num_sweeps=num_frames - 1,
        point_cloud_range=point_cloud_range,
        evaluation_distance_ranges=evaluation_distance_ranges))

evaluation_output_dir = (
    'outputs/racformer_company_front_50m_q200_f4_30k_train/evaluation/')
evaluation_profiles = ['car_only', 'main3']

# One rolling resume checkpoint plus one validation-best checkpoint.
checkpoint_config = dict(interval=2, max_keep_ckpts=1, save_last=True)
eval_config = dict(
    interval=2,
    save_best='company/car_3D_AP@0.5',
    rule='greater')
