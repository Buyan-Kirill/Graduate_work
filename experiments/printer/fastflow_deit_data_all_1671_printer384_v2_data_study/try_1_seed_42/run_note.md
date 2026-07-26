# deit_data_all_1671 / seed 42

Status at creation: planned train + calibration; test locked.
Role: all eligible normal train tiles without a volume cap.
Split: printer_384_v2_data_study.
Manifest SHA-256: `972f3456771e7190aaa8b1101f4de7b19d96ac19a911d004a65fdc62a7909a39`.
Pipeline SHA-256: `b83e964cf3a98e35d378d7f7c7769a2732f4c2eb56eebcdd8de06d6e878e811f`.
Git commit: `d7a03c6d1a6af92a9f6122f560a45e8624150f18`.
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
- augmentation: none
- early_stopping_patience: None
- recipe_source_git_commit: None
- train_sampling_rank: train_rank_date_balanced
- train_target: 1671

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Outcome

Validation report passed. Calibration primary ROC AUC is
`0.8882954545454544`, only `+0.0151136363636362` versus baseline and well
below date-balanced 500 (`0.9309090909090909`). Object/source ROC AUC are
`0.935 / 0.8`; threshold FNR is `0.5954545454545455`. The run has the lowest
normal-val loss but weak ranking, so optimization loss did not predict anomaly
quality. Test was not read.
