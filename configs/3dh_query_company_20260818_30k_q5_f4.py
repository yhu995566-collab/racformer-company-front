_base_ = ['./3dh_query_company_20260818_q5.py']

# Production-scale rerun of the best long-range query-layout ablation.
# The dataset root is supplied through RACFORMER_COMPANY_PROCESSED_ROOT by the
# guarded launcher.  Keeping a distinct config stem isolates checkpoints from
# the original small-data Q5 run.
evaluation_output_dir = (
    'outputs/3dh_query_company_20260818_30k_q5_f4/evaluation/')
