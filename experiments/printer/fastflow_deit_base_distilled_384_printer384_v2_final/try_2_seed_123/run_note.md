# deit_base_distilled_384 / seed 123

Status at creation: planned train + calibration; test locked.
Role: transformer candidate using the MVTec-tested recipe.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `47ac8a753676ee846cb4ce2ccb2d7505ca63ba706dd110777b773f421ee528c3`.
Git commit: `1293c34133af2a1333709cd01f2d3a3c6266382c`.
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
- Training duration: 1205.1666481 seconds; full stage elapsed: 1208.3400101
  seconds.
- Best epoch: 40 of 40; best normal-validation loss: 98305.18627929688.
- Gradient clipping was active in 100% of batches. Mean pre-clip norm averaged
  across epochs was 403047.908547852; maximum observed batch norm was
  2055065.375.
- Selected top-k: 147456 / 147456 pixels (1.0).
- Calibration source-group-balanced tile ROC AUC: 0.9553409090909091.
- Calibration object ROC AUC: 0.9750000000000001; source-image ROC AUC: 0.88.
- Versus paired ResNet18-384 seed 123, primary ROC AUC is
  +0.0185227272727273. This reverses the seed-42 difference of
  -0.0467045454545455.
- DeiT calibration scores across seeds have Pearson/Spearman correlations
  0.920909975872317 / 0.9157059756742465, lower than ResNet's
  0.9773891678308602 / 0.9738762559492332.

The figures were visually checked against source CSV/JSON. Because backbone
ordering reverses across seeds and DeiT variance is much larger than ResNet,
the predeclared third paired seed is required before freezing configurations.
No test data was used for this decision.

## Locked test outcome

- Executed once after calibration freeze; selection matches
  `calibration_selection.json` (full map and threshold
  `-0.5468745648860932`).
- Source-group-balanced tile ROC AUC: `0.9751286758089479`; per-run 95%
  hierarchical-bootstrap interval: `[0.9408946118129792,
  0.9967712512355369]`.
- Tile threshold counts: `FP=3`, `FN=7`, `TN=50`, `TP=38`.
- Object-max threshold counts: `FP=3`, `FN=0`, `TN=50`, `TP=7`.
- Source-image-max threshold counts: `FP=3`, `FN=0`, `TN=6`, `TP=7`.
- Paired primary difference versus ResNet seed 123 is
  `+0.0423364868943100`. DeiT makes 10 tile errors versus ResNet's 14.

This seed favors DeiT in ranking and total threshold errors; the predeclared
seed 2025 remains necessary for the final paired conclusion.
