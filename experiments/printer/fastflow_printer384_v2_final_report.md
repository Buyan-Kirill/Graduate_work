# FastFlow printer384 v2 final report

## Short conclusion

The frozen test does not reproduce the old DeiT ranking deficit. DeiT has
higher source-group-balanced tile ROC AUC on all three paired seeds and a mean
advantage of `+0.044121`. The paired 95% hierarchical-bootstrap interval
`[-0.034323, 0.146068]` includes zero, so superiority is not statistically
established with only 9 normal and 7 anomalous test source groups.

The result depends on the operating unit. DeiT fails the predeclared per-tile
error tolerance on two seeds because the q95-normal threshold has high anomaly
FNR. At object/source max level, DeiT has fewer or equal errors for every seed.

## Frozen setup

- Data: `printer_384_v2_final`, manifest SHA-256
  `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
- Independent groups: calibration `5 good + 5 anomaly`; test
  `9 good + 7 anomaly`.
- Both primary systems use `384x384`, no train augmentation, frozen pretrained
  Anomalib feature extractors and seeds `42, 123, 2025`.
- ResNet18: 30 epochs, 10 flow steps, LR `1e-3`, weight decay `1e-3`, batch 8.
- DeiT distilled base: 40 epochs, 8 flow steps, hidden ratio `0.5`, LR `3e-5`,
  weight decay `1e-5`, batch 10, gradient clip `10.0`.
- Top-k and threshold were selected on calibration only. Test was evaluated
  once after freeze commit `b2893c7`; no post-test tuning was performed.

## Calibration evidence

| Seed | ResNet AUC | DeiT AUC | DeiT - ResNet |
|---:|---:|---:|---:|
| 42 | 0.919886 | 0.873182 | -0.046705 |
| 123 | 0.936818 | 0.955341 | +0.018523 |
| 2025 | 0.918750 | 0.942614 | +0.023864 |

Calibration means are ResNet `0.925152` and DeiT `0.923712`. DeiT has much
higher sample SD (`0.044221` versus `0.010120`). Full-map aggregation
(`147456 / 147456` pixels) is the primary-AUC argmax for every primary run.

The clipping ablation lowers DeiT normal-validation likelihood loss by
`20.7%`, but changes primary calibration AUC by only `+0.002159`, leaves
source AUC unchanged and worsens threshold balanced accuracy. Clipping is not
the cause of the ranking gap seen on seed 42.

Calibration sources:

- `printer384_v2_final_calibration_comparison/calibration_comparison.md`
- `fastflow_printer384_v2_findings.md`

## Locked test evidence

| Seed | ResNet AUC | DeiT AUC | Delta | Tile errors R/D | Object errors R/D | Source errors R/D |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.905938 | 0.927615 | +0.021676 | 14 / 28 | 7 / 5 | 4 / 4 |
| 123 | 0.932792 | 0.975129 | +0.042336 | 14 / 10 | 5 / 3 | 4 / 3 |
| 2025 | 0.877060 | 0.945411 | +0.068351 | 17 / 27 | 7 / 2 | 4 / 2 |

- ResNet mean/sample SD: `0.9052633181884883 / 0.02787243570579096`.
- DeiT mean/sample SD: `0.9493846080013881 / 0.02400504914096937`.
- Mean paired delta: `+0.0441212898128999`.
- Paired hierarchical-bootstrap 95% CI:
  `[-0.03432287729906761, 0.14606808535379953]`.
- Total errors across seeds: tile ResNet/DeiT `45/65`, object `19/10`, source
  `12/9`.

Machine-readable sources:

- `printer384_v2_final_test_comparison_v2/fastflow_printer384_v2_final_auto_comparison.json`
- `printer384_v2_final_test_comparison_v2/fastflow_printer384_v2_final_auto_comparison_per_seed.csv`
- `printer384_v2_final_test_comparison_v2/fastflow_printer384_v2_final_auto_comparison_tile_scores.csv`
- `printer384_v2_final_test_comparison_v2/fastflow_printer384_v2_final_auto_comparison_object_scores.csv`
- `printer384_v2_final_test_comparison_v2/fastflow_printer384_v2_final_auto_comparison_source_scores.csv`

## Supported explanations

1. The old printer DeiT deficit is not an intrinsic deterministic backbone
   limitation: under the frozen protocol DeiT test ranking is higher on every
   seed. The earlier result was not reproducible after fixing the split,
   evaluation and top-k logic. Because several factors changed, their exact
   causal contributions cannot be separated retrospectively.
2. Small calibration composition and training seed interaction can reverse
   the apparent ordering. Calibration favored ResNet on seed 42, while test
   favored DeiT on all seeds.
3. The remaining practical weakness is threshold calibration, not ranking.
   The fixed 95th normal-score percentile controls FPR but does not optimize
   anomaly recall. High FNR was already visible on calibration and persists on
   test. Changing that rule now would be test tuning.
4. The operational unit matters. Many anomalous tiles fall below threshold,
   but object/source max often retains one strong detected tile. This explains
   why DeiT tile errors are worse while object/source errors are better.

## Limits and next evidence

- The confidence interval includes zero; do not claim statistically proven
  DeiT superiority.
- Multiple tiles are correlated within source captures. Effective test size is
  16 source groups, not 98 independent samples.
- There are no pixel masks, so localization accuracy is unmeasured.
- Do not tune on the current test. Collect more independent calibration
  captures, predeclare an application-specific threshold objective, and
  confirm it once on a new future-session holdout.
- Manual reproduction and inspection:
  `../../code/FastFlow_printer_final_reproduction.ipynb`.
