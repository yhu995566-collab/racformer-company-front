_base_ = ['./racformer_company_front_velocity_v2.py']

# Four-frame velocity-v2 experiment:
#   current frame + 3 historical sweeps.
# Keep the dataset, model hyperparameters, image resolution, radar velocity
# channels, and output isolation consistent with the 8-frame velocity-v2 run.
num_frames = 4
num_cams = 1

point_cloud_range = [0.0, -20.0, -3.0, 200.0, 20.0, 3.0]
grid_config = {
    'x': [0.0, 200.0, 0.5],
    'y': [-20.0, 20.0, 0.5],
    'z': [-3.0, 3.0, 6.0],
    'depth': [1.0, 205.0, 96.0],
    'rcs': [-64, 64, 64]
}
ida_aug_conf = {
    'resize_lim': (1.0, 1.1),
    'final_dim': (256, 640),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 480,
    'W': 640,
    'rand_flip': True,
}
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

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
         keys=[
             'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_depth',
             'radar_depth', 'radar_rcs', 'radar_points'
         ],
         meta_keys=(
             'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img',
             'img_timestamp', 'intrinsics'))
]

test_pipeline = [
    dict(type='LoadFrontCameraSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadCompanyRadarSweeps', sweeps_num=num_frames - 1,
         load_dim=7, roi=point_cloud_range),
    dict(type='LoadCompanyLidarPoints', load_dim=5, use_dim=5,
         roi=point_cloud_range),
    dict(type='FrontViewFilter', roi=point_cloud_range),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf,
         training=False),
    dict(type='PointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=num_cams),
    dict(type='RadarPointToMultiViewDepth', downsample=1,
         grid_config=grid_config, num_cams=num_cams, test_mode=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(640, 480),
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

model = dict(
    pts_bbox_head=dict(
        transformer=dict(num_frames=num_frames)))

data = dict(
    train=dict(pipeline=train_pipeline, num_sweeps=num_frames - 1),
    val=dict(pipeline=test_pipeline, num_sweeps=num_frames - 1),
    test=dict(pipeline=test_pipeline, num_sweeps=num_frames - 1))

evaluation_output_dir = (
    'outputs/racformer_company_front_velocity_v2_f4/evaluation/')

checkpoint_config = dict(interval=2, max_keep_ckpts=3)
