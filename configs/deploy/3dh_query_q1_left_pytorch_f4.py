_base_ = ['../3dh_query_q1.py']

# Deployment consumes synchronized camera/radar frames directly. Empty dataset
# pipelines prevent LiDAR loading, GT-depth generation, filtering, formatting,
# or DataContainer collation from entering the export path.
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
