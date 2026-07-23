# resnet18_384 / seed 2025

Status at creation: planned train + calibration; test locked.
Role: resolution-matched ResNet18 control for DeiT-384.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `47ac8a753676ee846cb4ce2ccb2d7505ca63ba706dd110777b773f421ee528c3`.
Git commit: `6589cdd2f79f68a9c5f1c4d826f5ed2c2fba668f`.
Git worktree dirty: `False`.

Training parameters:

- input: 384x384
- epochs: 30
- flow_steps: 10
- learning_rate: 0.001
- weight_decay: 0.001
- batch_size: 8
- grad_clip_norm: None
- augmentation: none

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Verified outcome

- Status: completed train + calibration; source validation passed; test was
  not read.
- Training duration: 625.3525914 seconds; full stage elapsed: 627.3981291
  seconds.
- Best epoch: 30 of 30; best normal-validation loss: -2965243.3.
- Selected top-k: 147456 / 147456 pixels (1.0).
- Calibration source-group-balanced tile ROC AUC: 0.91875.
- Calibration object ROC AUC: 0.96; source-image ROC AUC: 0.92.

Training, top-k and score-distribution figures were visually checked against
their CSV/JSON sources. Full-map top-k is now repeated across all three ResNet
seeds. Proceed to the paired DeiT seed 2025; test remains untouched.
