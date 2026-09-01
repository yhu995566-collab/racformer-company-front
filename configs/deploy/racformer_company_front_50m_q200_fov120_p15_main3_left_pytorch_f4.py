import os as _os

_base_ = ['../racformer_company_front_50m_q200_f4_30k_train.py']

# Frozen deployment geometry for the final Main3 FOV experiment.  Do not use
# the environment-driven tuning config here: an unset environment variable
# would silently restore the old rectangular query layout.
num_frames = 4
num_cams = 1
horizontal_fov_deg = 120.0
query_distance_power = 1.5
class_names = ['car', 'truck', 'bicycle']

deploy_data_root = _os.path.abspath(_os.environ.get(
    'RACFORMER_DEPLOY_DATA_ROOT',
    '/mnt/diskNvme1/hyh/data/'
    'company_20260818_30k_front50_q200_f4/processed_trainval_v1')) + '/'
deploy_ann_file = _os.path.abspath(_os.environ.get(
    'RACFORMER_DEPLOY_ANN_FILE',
    deploy_data_root + 'custom_infos_val_sweep.pkl'))

point_cloud_range = [0.0, -20.0, -3.0, 50.0, 20.0, 3.0]
ida_aug_conf = {
    'resize_lim': (0.334, 0.38),
    'final_dim': (256, 640),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080,
    'W': 1920,
    'rand_flip': False,
}

model = dict(pts_bbox_head=dict(
    num_classes=3,
    code_weights=[2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    query_init_mode='front_fov_grid',
    query_distance_power=query_distance_power,
    horizontal_fov_deg=horizontal_fov_deg,
    transformer=dict(num_frames=num_frames, num_classes=3),
    bbox_coder=dict(num_classes=3),
    loss_bbox=dict(type='L1Loss', loss_weight=0.25)))

# Deployment reads synchronized raw frames directly.  The dedicated
# DeploymentPreprocessor performs deterministic image resize/crop and applies
# rectangle-intersect-FOV filtering after radar points are in ego coordinates.
data = dict(
    val=dict(
        data_root=deploy_data_root, ann_file=deploy_ann_file,
        pipeline=[], classes=class_names, num_sweeps=num_frames - 1,
        horizontal_fov_deg=horizontal_fov_deg),
    test=dict(
        data_root=deploy_data_root, ann_file=deploy_ann_file,
        pipeline=[], classes=class_names, num_sweeps=num_frames - 1,
        horizontal_fov_deg=horizontal_fov_deg))

deployment = dict(
    camera='left',
    num_cams=num_cams,
    num_frames=num_frames,
    source_image_size=(1920, 1080),
    network_image_size=(640, 256),
    horizontal_fov_deg=horizontal_fov_deg,
    roi_mode='rectangle_fov_intersection',
    radar_point_fields=['x', 'y', 'z', 'rcs', 'vx', 'vy', 'time_lag'],
    radar_points_in_ego=True,
    image_color_order='BGR',
    image_dtype='uint8')
