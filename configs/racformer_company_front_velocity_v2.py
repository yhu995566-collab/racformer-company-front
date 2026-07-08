_base_ = ['./racformer_company_front.py']

# Keep the velocity-enabled company dataset physically separate from earlier
# converted datasets.  `data/company_dataset_velocity_v2` is intended to be a
# symlink to the corresponding directory on the server data disk.
dataset_root = 'data/company_dataset_velocity_v2/processed/'

data = dict(
    train=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'custom_infos_train_sweep.pkl'),
    val=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'custom_infos_val_sweep.pkl'),
    test=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'custom_infos_test_sweep.pkl'))

# Validation predictions are also kept below this experiment's own output
# tree.  On the server, the top-level `outputs` directory can be a data-disk
# symlink without changing this config.
evaluation_output_dir = 'outputs/racformer_company_front_velocity_v2/evaluation/'

# Match checkpointing to the two-epoch validation cadence and limit disk use.
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
