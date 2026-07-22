# resnet18_384 / seed 42 calibration result

Source validation: passed.

- best epoch: 30
- best normal-val loss: -2937849.9875
- selected top-k: 147456 (1.000000)
- source-balanced tile ROC AUC: 0.919886
- object ROC AUC: 0.965000
- source-image ROC AUC: 0.880000
- balanced accuracy: 0.765909
- FPR: 0.050000
- FNR: 0.418182

Figures:

- `calibration_training_curves.png`
- `calibration_top_k_sweep.png`
- `calibration_score_distribution.png`

Machine-readable values and source hashes: `calibration_report.json`.
