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
| H1 | Matching ResNet18 input resolution from 256 to 384 materially changes printer calibration quality. | not supported on seed 42 | ResNet18-384 changed primary calibration ROC AUC by only `+0.005227` and source-image ROC AUC by `0.0` versus ResNet18-256. | Treat ResNet18-384 as the fair matched control, not as a demonstrated resolution improvement. Do not spend a run repeating ResNet18-256. |
| H2 | The MVTec-tested DeiT optimizer recipe trains stably on the printer dataset. | pending | DeiT loss/LR curves, best epoch, pre-clip gradient norms, clip fraction, seed consistency. | Run only after validating ResNet18-384 seed 42. |
| H3 | The main DeiT gap, if present, is representation/spatial-resolution related rather than optimizer failure. | pending | Stable DeiT likelihood training but consistently weaker calibration ranking across paired seeds; compare with the architecture fact below. | Do not change architecture before H2 is evaluated. |
| H4 | A selected top-k is a stable property rather than calibration overfit. | inconclusive | Both ResNet resolutions at seed 42 select the full map and rise toward it, but they share seed and calibration groups. | Check DeiT and second paired seed; never choose top-k from test. |

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

### Attempt 2: ResNet18-384, seed 42, try 1

- Status: failed before model construction; this does not count as a training
  run and provides no evidence for H1.
- Completed epochs/training batches: `0 / 0`.
- Calibration and locked test inference were not run.
- Cause: Anomalib created `FastflowModel(pre_trained=True)`, which calls `timm`;
  `timm` attempted Hugging Face metadata access through an unavailable local
  proxy before using cached weights.
- Verified recovery: both ResNet18 and DeiT Anomalib models constructed
  successfully from local cache under process-local `HF_HUB_OFFLINE=1`.
- Cached ResNet source: `timm/resnet18.a1_in1k`, snapshot `491b427b...`, cached
  `model.safetensors` SHA-256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`.
- Cached DeiT source: `timm/deit_base_distilled_patch16_384.fb_in1k`, snapshot
  `58d8039f...`, cached `model.safetensors` SHA-256
  `f739dfae2bf3fdd4ef415fdb015966c515b9adca7de703a08365fb349fea82ca`.

Sources:

- `fastflow_resnet18_384_printer384_v2_final/try_1_seed_42/failure.json`
- `fastflow_resnet18_384_printer384_v2_final/try_1_seed_42/execution_provenance.json`
- `fastflow_resnet18_384_printer384_v2_final/try_1_seed_42/run_note.md`

### Run 2: ResNet18-384, seed 42, preserved try 2

- Status: completed train + calibration; source validation passed; locked test
  inference was not run.
- Training duration: `626.8787307000002` seconds; full train/calibration stage
  elapsed by file timestamps: `628.9650617` seconds.
- Peak allocated CUDA memory: `1954.37451171875` MiB.
- Best epoch/loss: `30 / 30`, `-2937849.9875`.
- Selected top-k: full map, `147456 / 147456` (`1.0`).
- Calibration source-group-balanced tile ROC AUC: `0.9198863636363637`.
- Calibration object ROC AUC: `0.9650000000000001`.
- Calibration source-image ROC AUC: `0.8800000000000001`.
- Versus ResNet18-256 seed 42: primary ROC AUC `+0.005227272727272636`,
  object ROC AUC `+0.01000000000000012`, source-image ROC AUC `0.0`.
- Training, top-k and score-distribution plots were visually checked and are
  consistent with the source CSV/JSON. Top-k ranking rises toward the full map.

Sources:

- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/run_note.md`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/train_history.csv`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/training_summary.json`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/calibration_selection.json`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/calibration_metrics.json`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/calibration_report.json`

## Decision log

| Date | Evidence available | Decision | Reason |
|---|---|---|---|
| 2026-07-23 | Run 1 plus frozen split audit. | Run ResNet18-384 seed 42 before DeiT. | Isolate input-resolution effect and establish the matched CNN control. |
| 2026-07-23 | User review of experiment cost. | Do not repeat ResNet18-256; use two paired seeds for the primary systems and add a third pair only if unstable. | Avoid spending nine runs solely on reproducibility. |
| 2026-07-23 | Attempt 2 failed before training; both backbones then built from the existing local cache in offline mode. | Preserve failed `try_1`; retry ResNet18-384 seed 42 as `try_2` with process-local `HF_HUB_OFFLINE=1`. | Use the same Anomalib/timm model path as the notebooks without changing proxy/VPN settings or downloading different weights. |
| 2026-07-23 | Completed ResNet18-384 seed 42: primary calibration ROC AUC `0.919886`, only `+0.005227` over ResNet18-256; both select full-map top-k. | Proceed to DeiT-384 seed 42 with the frozen MVTec-tested recipe. | Resolution alone did not materially change this seed; the matched ResNet control is valid and no extra 256 run is justified. |

## Update checklist after each attempt

1. Verify metric/report values against source CSV/JSON and hashes.
2. Update the run's `run_note.md` with outcome and decision.
3. Update `printer384_v2_experiment_ledger.csv`, including failed runs.
4. Update hypothesis status here as `supported`, `not supported`, or
   `inconclusive`; retain contradictory evidence.
5. Commit compact artifacts and this register before starting another run.
