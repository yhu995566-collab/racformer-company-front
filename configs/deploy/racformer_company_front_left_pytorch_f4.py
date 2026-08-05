_base_ = ['../racformer_company_front_velocity_v2_f4.py']

# Deployment reads raw synchronized frames directly. Keep dataset pipelines
# empty so LiDAR, GT depth, filtering, formatting, and DataContainer collation
# cannot be reintroduced during ONNX fixture generation.
data = dict(
    val=dict(pipeline=[]),
    test=dict(pipeline=[]))

deployment = dict(
    camera='left',
    num_cams=1,
    num_frames=4,
    radar_point_fields=['x', 'y', 'z', 'rcs', 'vx', 'vy', 'time_lag'],
    radar_points_in_ego=True,
    image_color_order='BGR',
    image_dtype='uint8')
