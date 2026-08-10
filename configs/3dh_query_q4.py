_base_ = ['./3dh_query_company_front_350m_f4.py']

# Q4: more longitudinal layers, 45 longitudinal x 20 lateral, 900 queries.
model = dict(pts_bbox_head=dict(
    num_query=900,
    num_clusters=45,
    query_distance_power=2.0,
    transformer=dict(num_ray=20)))

evaluation_output_dir = (
    'outputs/3dh_query_q4_far20_45x20/evaluation/')
