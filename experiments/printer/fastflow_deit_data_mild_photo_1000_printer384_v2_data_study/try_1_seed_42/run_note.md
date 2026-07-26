# deit_data_mild_photo_1000 / seed 42

Status at creation: planned train + calibration; test locked.
Role: mild photometric augmentation ablation at fixed train data.
Split: printer_384_v2_data_study.
Manifest SHA-256: `972f3456771e7190aaa8b1101f4de7b19d96ac19a911d004a65fdc62a7909a39`.
Pipeline SHA-256: `b83e964cf3a98e35d378d7f7c7769a2732f4c2eb56eebcdd8de06d6e878e811f`.
Git commit: `4faa2560add4daa0eaa8b1e0e5fe1d24b90dec21`.
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
- augmentation: mild_photometric
- early_stopping_patience: None
- recipe_source_git_commit: None
- train_sampling_rank: train_rank_date_balanced
- train_target: 1000

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Outcome

Failed before model construction and before the first training batch.
`timm` attempted Hugging Face metadata access through an unavailable proxy.
The exact pretrained weights were then confirmed to exist in the local cache
with process-local `HF_HUB_OFFLINE=1`. No proxy, VPN or Git settings were
changed. The real training retry uses a new `try_2` directory.
