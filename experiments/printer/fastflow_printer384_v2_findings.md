# FastFlow printer384 v2 evidence register

Last updated: 2026-07-23

This file is the central memory for tested hypotheses and experiment decisions.
It must be updated after every attempted run and before the next training run.
Numbers are included only with a path to their source artifact.

## Method guardrails

- Training/checkpoint selection uses train and disjoint normal validation loss.
- Top-k and threshold are selected only on calibration.
- No test image decoding, inference, score inspection, or test metric
  computation is allowed before the final configuration freeze.
- At most one alternative DeiT recipe may be tested after the baseline paired
  comparison. A poor locked-test result does not start tuning on the same test.
- `resnet18_256`, seed 42 is a historical reference. The primary paired
  comparison is `resnet18_384` versus `deit_base_distilled_384`.

## Hypothesis register

| ID | Hypothesis | Status | Evidence required | Current decision |
|---|---|---|---|---|
| H1 | Matching ResNet18 input resolution from 256 to 384 materially changes printer calibration quality. | pending | ResNet18-256 and ResNet18-384 seed-42 calibration reports; paired seeds if the difference affects the final comparison. | Run ResNet18-384 seed 42 next. |
| H2 | The MVTec-tested DeiT optimizer recipe trains stably on the printer dataset. | pending | DeiT loss/LR curves, best epoch, pre-clip gradient norms, clip fraction, seed consistency. | Run only after validating ResNet18-384 seed 42. |
| H3 | The main DeiT gap, if present, is representation/spatial-resolution related rather than optimizer failure. | pending | Stable DeiT likelihood training but consistently weaker calibration ranking across paired seeds; compare with the architecture fact below. | Do not change architecture before H2 is evaluated. |
| H4 | A selected top-k is a stable property rather than calibration overfit. | pending | Broad top-k performance plateau and similar selected region across paired seeds/backbones; source-group bootstrap/sensitivity. | Never choose top-k from test. |

Architecture fact relevant to H3: in the current Anomalib FastFlow
implementation, ResNet18 contributes three feature scales, while DeiT at 384
provides one 24x24 patch-token feature map. This fact makes H3 plausible for
small local defects, but does not establish it without experimental evidence.

## Completed observations

### Run 1: ResNet18-256, seed 42

- Status: completed train + calibration; locked test inference was not run.
- Best epoch: 14 of 14.
- Best normal-validation loss: `-976220.853125`.
- Selected top-k: `65536 / 65536` pixels (`1.0`).
- Calibration source-group-balanced tile ROC AUC: `0.914659090909091`.
- Calibration object ROC AUC: `0.955`.
- Calibration source-image ROC AUC: `0.8800000000000001`.
- The primary top-k curve rises toward the full-map score. This is one-run
  evidence only; it does not yet establish cross-seed top-k stability.

Sources:

- `fastflow_resnet18_256_printer384_v2_final/try_1_seed_42/run_note.md`
- `fastflow_resnet18_256_printer384_v2_final/try_1_seed_42/train_history.csv`
- `fastflow_resnet18_256_printer384_v2_final/try_1_seed_42/training_summary.json`
- `fastflow_resnet18_256_printer384_v2_final/try_1_seed_42/calibration_selection.json`
- `fastflow_resnet18_256_printer384_v2_final/try_1_seed_42/calibration_metrics.json`
- `fastflow_resnet18_256_printer384_v2_final/try_1_seed_42/calibration_report.json`

## Decision log

| Date | Evidence available | Decision | Reason |
|---|---|---|---|
| 2026-07-23 | Run 1 plus frozen split audit. | Run ResNet18-384 seed 42 before DeiT. | Isolate input-resolution effect and establish the matched CNN control. |
| 2026-07-23 | User review of experiment cost. | Do not repeat ResNet18-256; use two paired seeds for the primary systems and add a third pair only if unstable. | Avoid spending nine runs solely on reproducibility. |

## Update checklist after each attempt

1. Verify metric/report values against source CSV/JSON and hashes.
2. Update the run's `run_note.md` with outcome and decision.
3. Update `printer384_v2_experiment_ledger.csv`, including failed runs.
4. Update hypothesis status here as `supported`, `not supported`, or
   `inconclusive`; retain contradictory evidence.
5. Commit compact artifacts and this register before starting another run.
