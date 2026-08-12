_base_ = ['./racformer_company_front_100m_q300_f4.py']

# Corrected 100 m experiment. The legacy q300 checkpoint was trained with a
# 65 m polar decoder radius even though its Cartesian ROI extended to 100 m.
# This config must therefore be trained from scratch (or deliberately
# fine-tuned); it must not be paired with that legacy checkpoint as if the
# coordinate semantics were unchanged.
polar_radius = 100.0

model = dict(
    pts_bbox_head=dict(
        polar_radius=polar_radius,
        transformer=dict(polar_radius=polar_radius)))

evaluation_output_dir = (
    'outputs/racformer_company_front_100m_q300_r100_f4/evaluation/')
