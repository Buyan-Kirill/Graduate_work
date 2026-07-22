# deit_base_distilled_384_no_clip / seed 42

Status at creation: planned train + calibration; test locked.
Role: single-factor DeiT clipping ablation after saturated baseline clipping.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `47ac8a753676ee846cb4ce2ccb2d7505ca63ba706dd110777b773f421ee528c3`.
Git commit: `f84806251c7c8ca888b59b4984009a5fe99aea66`.
Git worktree dirty: `False`.

Training parameters:

- input: 384x384
- epochs: 40
- flow_steps: 8
- learning_rate: 3e-05
- weight_decay: 1e-05
- batch_size: 10
- grad_clip_norm: None
- augmentation: none

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Verified outcome

- Status: completed train + calibration; source validation passed; test was
  not read.
- Training duration: 1199.5594227 seconds; full stage elapsed: 1202.7186421
  seconds.
- Best epoch: 40 of 40; best normal-validation loss: 77184.96472167969.
- Selected top-k: 147456 / 147456 pixels (1.0).
- Calibration source-group-balanced tile ROC AUC: 0.875340909090909.
- Calibration object ROC AUC: 0.9299999999999999; source-image ROC AUC: 0.8.
- Versus the clipped DeiT seed-42 baseline, normal-validation loss is 20.7%
  lower, but primary ROC AUC changes by only +0.0021590909090908, object ROC
  AUC by +0.005, and source-image ROC AUC by 0.0. Threshold balanced accuracy
  changes from 0.7113636363636363 to 0.7022727272727272.
- Paired calibration scores are highly similar to the clipped baseline:
  Pearson correlation 0.9903035126593397 and Spearman correlation
  0.9881015335801163 over 61 images.

Interpretation: removing clipping materially improves normal likelihood but
does not materially close the anomaly-ranking gap to ResNet18-384. The tiny
primary-AUC increase is not sufficient to select this recipe from one small
calibration split, especially because tile AP and threshold performance are
worse. Do not repeat this diagnostic at seed 123. Continue the predeclared
paired-seed check with the original MVTec-tested DeiT baseline.
