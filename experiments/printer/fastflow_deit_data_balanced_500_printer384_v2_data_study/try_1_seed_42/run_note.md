# deit_data_balanced_500 / seed 42

Status at creation: planned train + calibration; test locked.
Role: nested 500-tile date-balanced train-volume ablation.
Split: printer_384_v2_data_study.
Manifest SHA-256: `972f3456771e7190aaa8b1101f4de7b19d96ac19a911d004a65fdc62a7909a39`.
Pipeline SHA-256: `b83e964cf3a98e35d378d7f7c7769a2732f4c2eb56eebcdd8de06d6e878e811f`.
Git commit: `2df31376cb0f32a817ac8ad65b9b09d214a49e93`.
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
- train_target: 500

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Outcome

Validation report passed. Calibration primary ROC AUC is
`0.9309090909090909`, versus baseline `0.8731818181818182`
(`+0.0577272727272727`). Object/source ROC AUC improve to `0.955 / 0.88`.
Full-map top-k is a smooth global maximum. The run passes the gate, but the
effect combines fewer tiles, stronger date balance and fewer optimizer steps.
Test was not read.
