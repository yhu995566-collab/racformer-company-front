_base_ = ['./3dh_query_company_front_350m_f4.py']

# Q3: strongly far-biased 30 longitudinal x 30 lateral, 900 queries.
model = dict(pts_bbox_head=dict(
    num_query=900,
    num_clusters=30,
    query_distance_power=2.0,
    transformer=dict(num_ray=30)))

evaluation_output_dir = (
    'outputs/3dh_query_q3_far20_30x30/evaluation/')
