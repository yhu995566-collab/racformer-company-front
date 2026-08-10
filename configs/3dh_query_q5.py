_base_ = ['./3dh_query_company_front_350m_f4.py']

# Q5: larger budget, 30 longitudinal x 40 lateral, 1200 queries.
model = dict(pts_bbox_head=dict(
    num_query=1200,
    num_clusters=30,
    query_distance_power=2.0,
    transformer=dict(num_ray=40)))

evaluation_output_dir = (
    'outputs/3dh_query_q5_far20_30x40/evaluation/')
