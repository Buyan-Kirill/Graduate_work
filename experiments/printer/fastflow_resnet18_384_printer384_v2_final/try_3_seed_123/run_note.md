# resnet18_384 / seed 123

Status at creation: planned train + calibration; test locked.
Role: resolution-matched ResNet18 control for DeiT-384.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `47ac8a753676ee846cb4ce2ccb2d7505ca63ba706dd110777b773f421ee528c3`.
Git commit: `d334f671f95f31352ee683c4ba5034e393ee3b02`.
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
- Training duration: 625.6792286 seconds; full stage elapsed: 627.7195591
  seconds.
- Best epoch: 30 of 30; best normal-validation loss: -2927802.25.
- Selected top-k: 147456 / 147456 pixels (1.0).
- Calibration source-group-balanced tile ROC AUC: 0.9368181818181818.
- Calibration object ROC AUC: 0.97; source-image ROC AUC: 0.92.
- Versus ResNet18-384 seed 42, primary/object/source ROC AUC changes are
  +0.0169318181818181, +0.005, and +0.04. Full-map top-k is unchanged.

Training, top-k and score-distribution figures were visually checked against
their CSV/JSON sources. This second seed confirms full-map top-k stability for
ResNet18 while showing moderate ranking variability. Proceed to the paired
DeiT baseline at seed 123; the locked test remains untouched.

## Locked test outcome

- Executed once after calibration freeze; selection matches
  `calibration_selection.json` (full map and threshold
  `-0.5344466358423233`).
- Source-group-balanced tile ROC AUC: `0.9327921889146379`; per-run 95%
  hierarchical-bootstrap interval: `[0.8461595670014037,
  0.9901832955404384]`.
- Tile threshold counts: `FP=3`, `FN=11`, `TN=50`, `TP=34`.
- Object-max threshold counts: `FP=3`, `FN=2`, `TN=50`, `TP=5`.
- Source-image-max threshold counts: `FP=2`, `FN=2`, `TN=7`, `TP=5`.

This is one predeclared seed result, not a standalone backbone conclusion.
