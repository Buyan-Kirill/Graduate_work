# resnet18_384 / seed 42

Status at creation: planned train + calibration; test locked.
Role: resolution-matched ResNet18 control for DeiT-384.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `35805411342e015e221c713dda22726b46ef013e25a431a6f2847bc0bad5e359`.
Git commit: `0390f3e0d08348449d5eb413f06953ba63e2c39f`.
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

## Outcome

Status: completed train + calibration. Locked test inference was not run.

- training duration: `626.8787307000002` seconds
- peak allocated CUDA memory: `1954.37451171875` MiB
- best epoch: `30 / 30`
- best normal-validation loss: `-2937849.9875`
- selected top-k: `147456 / 147456` pixels (`1.0`)
- calibration source-group-balanced tile ROC AUC: `0.9198863636363637`
- calibration object ROC AUC: `0.9650000000000001`
- calibration source-image ROC AUC: `0.8800000000000001`

Source validation passed in `calibration_report.json`. The saved training,
top-k and score-distribution plots were visually checked. Compared with the
ResNet18-256 seed-42 reference, the primary calibration ROC AUC changed by
only `+0.005227272727272636`; source-image ROC AUC did not change. This is
evidence against a material resolution-only gain on this seed, not a
cross-seed conclusion.
