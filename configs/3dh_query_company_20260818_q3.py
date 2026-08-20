_base_ = ['./3dh_query_company_20260818_f4.py']

model = dict(pts_bbox_head=dict(
    num_query=900,
    num_clusters=30,
    query_distance_power=2.0,
    transformer=dict(num_ray=30)))

evaluation_output_dir = (
    'outputs/3dh_query_company_20260818_q3/evaluation/')
