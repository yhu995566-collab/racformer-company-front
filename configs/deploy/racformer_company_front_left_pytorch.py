_base_ = ['../racformer_company_front_velocity_v2.py']

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
