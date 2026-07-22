# deit_base_distilled_384 / seed 42

Status at creation: planned train + calibration; test locked.
Role: transformer candidate using the MVTec-tested recipe.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `35805411342e015e221c713dda22726b46ef013e25a431a6f2847bc0bad5e359`.
Git commit: `20b4450b571f6c981f91c7b43f6973e039be2e67`.
Git worktree dirty: `False`.

Training parameters:

- input: 384x384
- epochs: 40
- flow_steps: 8
- learning_rate: 3e-05
- weight_decay: 1e-05
- batch_size: 10
- grad_clip_norm: 10.0
- augmentation: none

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Verified outcome

- Status: completed train + calibration; source validation passed; test was
  not read.
- Training duration: 1207.3457692 seconds; full stage elapsed: 1210.4920653
  seconds.
- Best epoch: 40 of 40; best normal-validation loss: 97379.98022460938.
- Gradient clipping was active in every training batch on every epoch. Mean
  pre-clip norm across epochs was 381344.400570312 and the maximum observed
  batch norm was 1839852.625, versus the configured threshold 10.0.
- Selected top-k: 147456 / 147456 pixels (1.0).
- Calibration source-group-balanced tile ROC AUC: 0.8731818181818182.
- Calibration object ROC AUC: 0.925; source-image ROC AUC: 0.8.
- Versus ResNet18-384 seed 42, the primary ROC AUC difference is
  -0.0467045454545455. Object and source-image differences are -0.04 and
  -0.08, respectively.

The likelihood curves are smooth and monotonically improving, so the run is
not numerically divergent. However, clipping at every batch means the
configured clipping is part of the optimizer dynamics rather than an
occasional safety guard. The next experiment is a single-factor no-clipping
ablation at seed 42. All other settings remain unchanged. This decision uses
calibration evidence only; the locked test remains untouched.
