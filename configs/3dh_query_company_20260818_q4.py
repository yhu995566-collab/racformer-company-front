_base_ = ['./3dh_query_company_20260818_f4.py']

model = dict(pts_bbox_head=dict(
    num_query=900,
    num_clusters=45,
    query_distance_power=2.0,
    transformer=dict(num_ray=20)))

evaluation_output_dir = (
    'outputs/3dh_query_company_20260818_q4/evaluation/')
