_base_ = ['../racformer_company_front_velocity_v2.py']

# The offline deployment checker only calls dataset.get_data_info(). Keeping
# these pipelines empty prevents accidental LiDAR loading, GT-depth generation,
# FrontViewFilter, formatting, or DataContainer collation.
data = dict(
    val=dict(pipeline=[]),
    test=dict(pipeline=[]))

# Deployment code reads this section directly. Model/checkpoint semantics are
# inherited unchanged from the trained velocity-v2 experiment.
deployment = dict(
    camera='left',
    num_cams=1,
    num_frames=8,
    radar_point_fields=['x', 'y', 'z', 'rcs', 'vx', 'vy', 'time_lag'],
    radar_points_in_ego=True,
    image_color_order='BGR',
    image_dtype='uint8')
