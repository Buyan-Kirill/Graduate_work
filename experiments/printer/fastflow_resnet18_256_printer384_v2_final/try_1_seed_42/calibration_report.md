# resnet18_256 / seed 42 calibration result

Source validation: passed.

- best epoch: 14
- best normal-val loss: -976220.853125
- selected top-k: 65536 (1.000000)
- source-balanced tile ROC AUC: 0.914659
- object ROC AUC: 0.955000
- source-image ROC AUC: 0.880000
- balanced accuracy: 0.747727
- FPR: 0.050000
- FNR: 0.454545

Figures:

- `calibration_training_curves.png`
- `calibration_top_k_sweep.png`
- `calibration_score_distribution.png`

Machine-readable values and source hashes: `calibration_report.json`.
