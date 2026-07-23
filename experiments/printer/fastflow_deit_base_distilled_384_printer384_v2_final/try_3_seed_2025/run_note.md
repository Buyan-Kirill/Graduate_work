# deit_base_distilled_384 / seed 2025

Status at creation: planned train + calibration; test locked.
Role: transformer candidate using the MVTec-tested recipe.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `47ac8a753676ee846cb4ce2ccb2d7505ca63ba706dd110777b773f421ee528c3`.
Git commit: `c7179a78fdac3099af7d23a23fdde2b43111cdb9`.
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
- Training duration: 1204.8147442 seconds; full stage elapsed: 1207.935336
  seconds.
- Best epoch: 40 of 40; best normal-validation loss: 81271.27047729492.
- Gradient clipping was active in 100% of batches. Mean pre-clip norm averaged
  across epochs was 196183.77545; maximum observed batch norm was 755359.5.
- Selected top-k: 147456 / 147456 pixels (1.0).
- Calibration source-group-balanced tile ROC AUC: 0.9426136363636364.
- Calibration object ROC AUC: 0.97; source-image ROC AUC: 0.84.
- Versus paired ResNet18-384 seed 2025, primary ROC AUC is
  +0.0238636363636364.
- Across seeds 42, 123 and 2025, primary mean/sample SD are
  0.9237121212121212 / 0.044221 for DeiT and
  0.9251515151515152 / 0.010120 for ResNet18. Mean paired DeiT-minus-ResNet
  difference is -0.0014393939393939; DeiT wins two of three seeds.

The figures were visually checked against source CSV/JSON. Full-map top-k is
stable across all six primary calibration runs. The mean ranking quality is
effectively tied on this small calibration split, but DeiT is more
seed-sensitive. No additional training is justified before the locked test.

## Locked test outcome

- Executed once after calibration freeze; selection matches
  `calibration_selection.json` (full map and threshold
  `-0.5377835601568223`).
- Source-group-balanced tile ROC AUC: `0.9454105858867765`; per-run 95%
  hierarchical-bootstrap interval: `[0.8901981704362658,
  0.9853617216117216]`.
- Tile threshold counts: `FP=1`, `FN=26`, `TN=52`, `TP=19`.
- Object-max threshold counts: `FP=1`, `FN=1`, `TN=52`, `TP=6`.
- Source-image-max threshold counts: `FP=1`, `FN=1`, `TN=8`, `TP=6`.
- Paired primary difference versus ResNet seed 2025 is
  `+0.0683510115482906`. DeiT makes 27 tile errors versus ResNet's 17, but
  only 2 object/source errors versus ResNet's 7/4.

This completes the predeclared locked-test matrix. No threshold or model
parameter is changed from these results.
