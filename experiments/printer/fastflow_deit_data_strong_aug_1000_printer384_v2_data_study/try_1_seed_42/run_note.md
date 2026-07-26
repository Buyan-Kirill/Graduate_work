# deit_data_strong_aug_1000 / seed 42

Status at creation: planned train + calibration; test locked.
Role: legacy strong augmentation ablation at fixed current DeiT recipe.
Split: printer_384_v2_data_study.
Manifest SHA-256: `972f3456771e7190aaa8b1101f4de7b19d96ac19a911d004a65fdc62a7909a39`.
Pipeline SHA-256: `b83e964cf3a98e35d378d7f7c7769a2732f4c2eb56eebcdd8de06d6e878e811f`.
Git commit: `b783ff0df4d1285da592d76ab8785de831c47094`.
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

## Outcome

Validation report passed. Calibration primary ROC AUC is
`0.9488636363636362`, versus `0.8731818181818182` for baseline
(`+0.0756818181818180`). Object/source ROC AUC improve from `0.925 / 0.8`
to `0.985 / 0.88`. The selected half-map top-k is supported by high adjacent
values rather than an isolated grid spike. The predeclared harm hypothesis is
contradicted on seed 42. The run passes the gate provisionally, but no
candidate is selected until all first-pass runs are complete. Test was not
read.
