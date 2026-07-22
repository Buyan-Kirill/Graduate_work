# resnet18_384 / seed 123 calibration result

Source validation: passed.

- best epoch: 30
- best normal-val loss: -2927802.25
- selected top-k: 147456 (1.000000)
- source-balanced tile ROC AUC: 0.936818
- object ROC AUC: 0.970000
- source-image ROC AUC: 0.920000
- balanced accuracy: 0.731818
- FPR: 0.050000
- FNR: 0.486364

Figures:

- `calibration_training_curves.png`
- `calibration_top_k_sweep.png`
- `calibration_score_distribution.png`

Machine-readable values and source hashes: `calibration_report.json`.
