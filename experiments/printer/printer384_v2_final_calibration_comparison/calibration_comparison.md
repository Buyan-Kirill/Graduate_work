# FastFlow printer384 v2 calibration comparison

Status: completed calibration-only comparison; locked test was not read.

Command (project `.venv`):

```powershell
.\.venv\Scripts\python.exe code\analyze_fastflow_printer_calibration.py `
  --configs resnet18_384 deit_base_distilled_384 `
  --seeds 42 123 2025 `
  --output-dir experiments\printer\printer384_v2_final_calibration_comparison
```

Primary metric: source-group-balanced tile ROC AUC.

| Seed | ResNet18-384 | DeiT-384 | DeiT - ResNet |
|---:|---:|---:|---:|
| 42 | 0.919886 | 0.873182 | -0.046705 |
| 123 | 0.936818 | 0.955341 | +0.018523 |
| 2025 | 0.918750 | 0.942614 | +0.023864 |

Aggregate facts:

- ResNet18 mean / sample SD: `0.9251515151515152 / 0.010119593070467087`.
- DeiT mean / sample SD: `0.9237121212121212 / 0.044220804052157184`.
- Mean paired difference: `-0.0014393939393939255`.
- Sample SD of paired difference: `0.039291624855006445`.
- DeiT wins: `2 / 3` seeds.
- The full `147456`-pixel map is the primary-metric argmax for every run.
  Some sweeps contain local non-monotonic steps; strict monotonicity is not
  claimed.

Interpretation: calibration supports similar mean ranking quality, not a
stable DeiT deficit. DeiT is materially more seed-sensitive on the small
5-normal-source / 5-anomaly-source calibration split. This is insufficient to
claim backbone equivalence; the next evidence is one frozen evaluation on the
untouched test for all three paired seeds.

Machine-readable sources are the CSV files in this directory. Per-run source
metrics, plots, configuration, provenance and checkpoints remain in each run
directory listed by `fastflow_printer384_v2_final_calibration_per_seed_auto.csv`.
