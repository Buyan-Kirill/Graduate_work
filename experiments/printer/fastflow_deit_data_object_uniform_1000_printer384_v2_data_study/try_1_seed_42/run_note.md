# deit_data_object_uniform_1000 / seed 42

Status at creation: planned train + calibration; test locked.
Role: 1000-tile object-uniform sampling ablation without date balancing.
Split: printer_384_v2_data_study.
Manifest SHA-256: `972f3456771e7190aaa8b1101f4de7b19d96ac19a911d004a65fdc62a7909a39`.
Pipeline SHA-256: `b83e964cf3a98e35d378d7f7c7769a2732f4c2eb56eebcdd8de06d6e878e811f`.
Git commit: `7901985de42411080992c0c86d9522a31269d651`.
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
- train_sampling_rank: train_rank_object_uniform
- train_target: 1000

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Outcome

Validation report passed. Calibration primary ROC AUC is
`0.906590909090909`, versus baseline `0.8731818181818182`
(`+0.0334090909090908`). Object/source ROC AUC are `0.95 / 0.88`.
The run passes the gate but trails date-balanced 500 by `0.0243181818181819`.
Full object/source coverage did not compensate for the strong date imbalance.
Test was not read.
