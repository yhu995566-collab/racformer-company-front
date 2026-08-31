"""Parametric, logged config for controlled 50 m Q200 tuning experiments."""

import os as _os

_base_ = ['./racformer_company_front_50m_q200_f4_30k_train.py']


def _env_bool(name, default):
    value = _os.environ.get(name, str(default)).strip().lower()
    if value not in ('0', '1', 'false', 'true', 'no', 'yes'):
        raise ValueError('{} must be a boolean, got {!r}'.format(name, value))
    return value in ('1', 'true', 'yes')


all_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
class_names = [item.strip() for item in _os.environ.get(
    'RACFORMER_TUNE_CLASSES', 'car,truck,bicycle').split(',') if item.strip()]
unknown_classes = sorted(set(class_names) - set(all_class_names))
if not class_names or unknown_classes:
    raise ValueError('invalid RACFORMER_TUNE_CLASSES: {}'.format(class_names))

num_frames = int(_os.environ.get('RACFORMER_TUNE_NUM_FRAMES', '4'))
if num_frames not in (1, 4):
    raise ValueError('RACFORMER_TUNE_NUM_FRAMES must be 1 or 4')
num_sweeps = num_frames - 1
rand_flip = _env_bool('RACFORMER_TUNE_RAND_FLIP', False)
horizontal_fov_deg = float(_os.environ.get(
    'RACFORMER_TUNE_HORIZONTAL_FOV_DEG', '0'))
if horizontal_fov_deg <= 0:
    horizontal_fov_deg = None
elif horizontal_fov_deg >= 180:
    raise ValueError('RACFORMER_TUNE_HORIZONTAL_FOV_DEG must be < 180')
total_epochs = int(_os.environ.get('RACFORMER_TUNE_EPOCHS', '12'))
eval_interval = int(_os.environ.get('RACFORMER_TUNE_EVAL_INTERVAL', '2'))
checkpoint_interval = int(_os.environ.get(
    'RACFORMER_TUNE_CHECKPOINT_INTERVAL', str(eval_interval)))
max_keep_ckpts = int(_os.environ.get('RACFORMER_TUNE_MAX_KEEP_CKPTS', '1'))
learning_rate = float(_os.environ.get('RACFORMER_TUNE_LR', '4e-4'))
query_distance_power = float(_os.environ.get(
    'RACFORMER_TUNE_QUERY_DISTANCE_POWER', '1.0'))
if query_distance_power < 1.0:
    raise ValueError('RACFORMER_TUNE_QUERY_DISTANCE_POWER must be >= 1.0')
bbox_loss_weight = float(_os.environ.get(
    'RACFORMER_TUNE_BBOX_LOSS_WEIGHT', '0.25'))
code_weights = [float(value) for value in _os.environ.get(
    'RACFORMER_TUNE_CODE_WEIGHTS',
    '2,2,1,1,1,1,1,1,1,1').split(',')]
if len(code_weights) != 10:
    raise ValueError('RACFORMER_TUNE_CODE_WEIGHTS must contain 10 values')

dataset_root = _os.path.abspath(_os.environ.get(
    'RACFORMER_TUNE_DATA_ROOT',
    '/mnt/diskNvme1/hyh/data/company_20260818_30k_tuning/proxy_main3')) + '/'
test_root = _os.path.abspath(_os.environ.get(
    'RACFORMER_COMPANY_TEST_ROOT',
    '/mnt/diskNvme1/hyh/data/'
    'company_20260818_30k_front50_q200_f4/processed_v3')) + '/'
train_ann_file = _os.path.abspath(_os.environ.get(
    'RACFORMER_TUNE_TRAIN_ANN_FILE',
    dataset_root + 'custom_infos_train_sweep.pkl'))
val_ann_file = _os.path.abspath(_os.environ.get(
    'RACFORMER_TUNE_VAL_ANN_FILE',
    dataset_root + 'custom_infos_val_sweep.pkl'))

point_cloud_range = [0.0, -20.0, -3.0, 50.0, 20.0, 3.0]
grid_config = {
    'x': [0.0, 50.0, 0.5], 'y': [-20.0, 20.0, 0.5],
    'z': [-3.0, 3.0, 6.0], 'depth': [1.0, 55.0, 96.0],
    'rcs': [-64, 64, 64],
}
ida_aug_conf = {
    'resize_lim': (0.334, 0.38), 'final_dim': (256, 640),
    'bot_pct_lim': (0.0, 0.0), 'rot_lim': (0.0, 0.0),
    'H': 1080, 'W': 1920, 'rand_flip': rand_flip,
}

train_pipeline = [
    dict(type='LoadFrontCameraSweeps', sweeps_num=num_sweeps),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False, with_label=False, with_bbox_depth=False),
    dict(type='LoadCompanyRadarSweeps', sweeps_num=num_sweeps, load_dim=7,
         roi=point_cloud_range),
    dict(type='LoadCompanyLidarPoints', load_dim=5, use_dim=5,
         roi=point_cloud_range),
    dict(type='FrontViewFilter', roi=point_cloud_range,
         horizontal_fov_deg=horizontal_fov_deg),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config,
         num_cams=1),
    dict(type='RadarPointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=1, test_mode=False),
    dict(type='RaCFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_depth',
               'radar_depth', 'radar_rcs', 'radar_points'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'lidar2img', 'img_timestamp', 'intrinsics')),
]

test_pipeline = [
    dict(type='LoadFrontCameraSweeps', sweeps_num=num_sweeps),
    dict(type='LoadCompanyRadarSweeps', sweeps_num=num_sweeps, load_dim=7,
         roi=point_cloud_range),
    dict(type='LoadCompanyLidarPoints', load_dim=5, use_dim=5,
         roi=point_cloud_range),
    dict(type='FrontViewFilter', roi=point_cloud_range,
         horizontal_fov_deg=horizontal_fov_deg),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config,
         num_cams=1),
    dict(type='RadarPointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=1, test_mode=True),
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

model = dict(pts_bbox_head=dict(
    num_classes=len(class_names),
    code_weights=code_weights,
    query_distance_power=query_distance_power,
    query_init_mode=('front_fov_grid' if horizontal_fov_deg else 'front_grid'),
    horizontal_fov_deg=(horizontal_fov_deg or 120.0),
    transformer=dict(num_frames=num_frames, num_classes=len(class_names)),
    bbox_coder=dict(num_classes=len(class_names)),
    loss_bbox=dict(type='L1Loss', loss_weight=bbox_loss_weight)),
    train_cfg=dict(pts=dict(assigner=dict(
        reg_cost=dict(type='BBox3DL1Cost', weight=bbox_loss_weight)))))

data = dict(
    train=dict(data_root=dataset_root,
               ann_file=train_ann_file,
               pipeline=train_pipeline, classes=class_names,
               num_sweeps=num_sweeps,
               horizontal_fov_deg=horizontal_fov_deg),
    val=dict(data_root=dataset_root,
             ann_file=val_ann_file,
             pipeline=test_pipeline, classes=class_names,
             num_sweeps=num_sweeps,
             horizontal_fov_deg=horizontal_fov_deg),
    test=dict(data_root=test_root,
              ann_file=test_root + 'custom_infos_test_sweep.pkl',
              pipeline=test_pipeline, classes=class_names,
              num_sweeps=num_sweeps,
              horizontal_fov_deg=horizontal_fov_deg))

optimizer = dict(lr=learning_rate)
checkpoint_config = dict(interval=checkpoint_interval,
                         max_keep_ckpts=max_keep_ckpts,
                         save_last=True)
eval_config = dict(interval=eval_interval,
                   save_best=('company/car_3D_AP@0.5'
                              if class_names == ['car']
                              else 'company/3D_mAP@0.5'),
                   rule='greater')
if class_names == ['car']:
    evaluation_profiles = ['car_only']
elif class_names == ['car', 'truck', 'bicycle', 'pedestrian']:
    evaluation_profiles = ['car_only', 'main3', 'main4']
else:
    evaluation_profiles = ['car_only', 'main3']
evaluation_output_dir = 'outputs/racformer_company_50m_q200_tune/evaluation/'
