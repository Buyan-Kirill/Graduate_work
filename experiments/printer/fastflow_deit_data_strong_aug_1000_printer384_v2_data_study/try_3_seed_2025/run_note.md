# deit_data_strong_aug_1000 / seed 2025

Status at creation: planned train + calibration; test locked.
Role: legacy strong augmentation ablation at fixed current DeiT recipe.
Split: printer_384_v2_data_study.
Manifest SHA-256: `972f3456771e7190aaa8b1101f4de7b19d96ac19a911d004a65fdc62a7909a39`.
Pipeline SHA-256: `b83e964cf3a98e35d378d7f7c7769a2732f4c2eb56eebcdd8de06d6e878e811f`.
Git commit: `3f4995b3bb08819acf7535dcc064b70adc6f96f6`.
Git worktree dirty: `False`.

Training parameters:

- input: 384x384
- epochs: 40
- flow_steps: 8
- hidden_ratio: 0.5
- learning_rate: 3e-05
- weight_decay: 1e-05
- eta_min: 1e-06
- batch_size: 10
- grad_clip_norm: 10.0
- augmentation: legacy_deit_try1
- early_stopping_patience: None
- recipe_source_git_commit: None
- train_sampling_rank: train_rank_date_balanced
- train_target: 1000

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Result

- Validation status: `passed`.
- Best epoch: `40`.
- Best normal-val loss: `134885.1240234375`.
- Training duration: `1437.7805985` seconds.
- Peak allocated CUDA memory: `1277.873046875 MiB`.
- Selected top-k: `147456` pixels (full map).
- Calibration primary ROC AUC: `0.9380681818181817`.
- Paired baseline primary ROC AUC: `0.9426136363636364`.
- Paired delta: `-0.0045454545454547`.
- Object/source-image ROC AUC: `0.99 / 0.92`.
- Threshold balanced accuracy/FPR/FNR:
  `0.8386363636363636 / 0.05 / 0.2727272727272727`.
- Calibration rows are identical to the paired baseline.
- Test was not read.
