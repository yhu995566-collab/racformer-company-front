_base_ = ['./3dh_query_company_front_350m_f4.py']

# Q1: uniform 30 longitudinal x 30 lateral, 900 total queries.
model = dict(pts_bbox_head=dict(
    num_query=900,
    num_clusters=30,
    query_distance_power=1.0,
    transformer=dict(num_ray=30)))

evaluation_output_dir = (
    'outputs/3dh_query_q1_uniform_30x30/evaluation/')
