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
| H2 | The MVTec-tested DeiT optimizer recipe trains stably on the printer dataset. | supported for numerical stability; clipping is unnecessary for stability | Removing clipping lowers final normal-val loss by `20.7%` without divergence, but changes primary calibration AUC by only `+0.002159`; clipped/no-clip score Spearman correlation is `0.988102`. | Do not repeat no-clip at seed 123. Use the original recipe for the predeclared paired-seed baseline and treat clipping as not causal for the ranking gap. |
| H3 | The main DeiT gap, if present, is representation/spatial-resolution related rather than optimizer failure. | not supported as a stable calibration gap; seed sensitivity remains | Across three paired seeds, DeiT-minus-ResNet differences are `-0.046705`, `+0.018523`, and `+0.023864`; mean difference is `-0.001439`. DeiT sample SD is `0.044221` versus ResNet `0.010120`. | Do not add architecture or optimizer variants before test. The calibration evidence supports similar mean ranking with substantially less stable DeiT initialization, not deterministic transformer inferiority. |
| H4 | A selected top-k is a stable property rather than calibration overfit. | supported across all three paired seeds | The full `147456`-pixel map is the primary-AUC argmax in all six primary runs; the no-clip diagnostic selects it too. Some sweeps have local non-monotonic steps, so only endpoint optimality is claimed. | Freeze full-map aggregation for both backbones and never choose top-k from test. |

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

### Run 3: DeiT-384, seed 42

- Status: completed train + calibration; source validation passed; locked test
  inference was not run.
- Training duration: `1207.3457691999993` seconds; full train/calibration stage
  elapsed by file timestamps: `1210.4920653` seconds.
- Peak allocated CUDA memory: `1277.873046875` MiB.
- Best epoch/loss: `40 / 40`, `97379.98022460938`.
- Train and normal-validation likelihood losses decrease smoothly. This rules
  out numerical divergence, but the endpoint best epoch does not establish a
  plateau.
- Gradient clipping at norm `10.0` was active in `100%` of batches in every
  epoch. Across epochs, mean pre-clip norm averaged `381344.400570312`; the
  maximum observed batch norm was `1839852.625`.
- Selected top-k: full map, `147456 / 147456` (`1.0`). The primary, object and
  tile top-k curves all improve toward the full map.
- Calibration source-group-balanced tile ROC AUC: `0.8731818181818182`.
- Calibration object ROC AUC: `0.925`.
- Calibration source-image ROC AUC: `0.8`.
- Versus ResNet18-384 seed 42: primary ROC AUC `-0.0467045454545455`,
  object ROC AUC `-0.04000000000000015`, source-image ROC AUC
  `-0.08000000000000007`.
- The figures were visually checked against CSV/JSON. Score overlap is largest
  for anomaly source `2025-09-10_15-20-02_L0122_1`; this is descriptive error
  analysis on calibration, not a reason to change labels or tune on test.

Interpretation: the frozen MVTec recipe is numerically stable but its clipping
threshold is not acting as an occasional guard on this dataset. Because every
batch is clipped, optimizer behavior remains a concrete confound and H3 cannot
yet be assigned to representation alone.

Sources:

- `fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42/run_note.md`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42/train_history.csv`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42/training_summary.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42/calibration_selection.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42/calibration_metrics.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42/calibration_report.json`

### Run 4: DeiT-384 no-clipping ablation, seed 42

- Status: completed train + calibration; source validation passed; locked test
  inference was not run.
- Training duration: `1199.5594227000001` seconds; full train/calibration stage
  elapsed by file timestamps: `1202.7186421` seconds.
- Peak allocated CUDA memory: `1277.87255859375` MiB.
- Best epoch/loss: `40 / 40`, `77184.96472167969`.
- The only behavioral config change from Run 3 is
  `grad_clip_norm: 10.0 -> None`; name/tag differ only for provenance.
- Versus clipped DeiT, final normal-val loss is `20195.0155029297` lower
  (`20.7%`), confirming that clipping materially constrained likelihood
  optimization.
- Selected top-k remains full map, `147456 / 147456` (`1.0`), and the
  endpoint is the primary-AUC argmax.
- Calibration source-group-balanced tile ROC AUC: `0.875340909090909`, only
  `+0.0021590909090908` versus clipped DeiT and `-0.0445454545454547` versus
  ResNet18-384.
- Object/source-image ROC AUC: `0.93 / 0.80`; differences versus clipped DeiT
  are `+0.005 / 0.0`.
- Threshold balanced accuracy is `0.7022727272727272`, down from
  `0.7113636363636363` with clipping. Tile ROC AUC/AP also decrease from
  `0.895238/0.794407` to `0.888095/0.771031`.
- Clipped/no-clip calibration scores have Pearson correlation `0.990304` and
  Spearman correlation `0.988102` across the same 61 images.
- Training, top-k and score-distribution plots were visually checked and are
  consistent with source CSV/JSON.

Interpretation: clipping was not required for stability and strongly affected
the likelihood scale, but it did not cause the anomaly-ranking deficit. The
small primary-AUC increase is mixed with worse AP/threshold evidence and is
too small to select no-clip from a 5+5-source calibration set. This diagnostic
will not be repeated at seed 123.

Sources:

- `fastflow_deit_base_distilled_384_no_clip_printer384_v2_final/try_1_seed_42/run_note.md`
- `fastflow_deit_base_distilled_384_no_clip_printer384_v2_final/try_1_seed_42/train_history.csv`
- `fastflow_deit_base_distilled_384_no_clip_printer384_v2_final/try_1_seed_42/training_summary.json`
- `fastflow_deit_base_distilled_384_no_clip_printer384_v2_final/try_1_seed_42/calibration_selection.json`
- `fastflow_deit_base_distilled_384_no_clip_printer384_v2_final/try_1_seed_42/calibration_metrics.json`
- `fastflow_deit_base_distilled_384_no_clip_printer384_v2_final/try_1_seed_42/calibration_report.json`

### Run 5: ResNet18-384, seed 123

- Status: completed train + calibration; source validation passed; locked test
  inference was not run.
- Training/full-stage duration: `625.6792286 / 627.7195591` seconds.
- Peak allocated CUDA memory: `1954.37451171875` MiB.
- Best epoch/loss: `30 / 30`, `-2927802.25`.
- Selected top-k remains full map, `147456 / 147456` (`1.0`). The primary,
  object and tile top-k curves rise toward the selected endpoint.
- Calibration source-group-balanced tile ROC AUC: `0.9368181818181818`.
- Calibration object/source-image ROC AUC: `0.97 / 0.92`.
- Versus seed 42, primary/object/source ROC AUC changes are
  `+0.0169318181818181 / +0.005 / +0.04`.
- Figures were visually checked against source CSV/JSON.

Interpretation: the second ResNet seed confirms full-map top-k stability and
shows moderate ranking variance. The paired DeiT seed 123 is required before
comparing backbone means or deciding whether a third seed is justified.

Sources:

- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/run_note.md`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/train_history.csv`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/training_summary.json`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/calibration_selection.json`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/calibration_metrics.json`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/calibration_report.json`

### Run 6: DeiT-384, seed 123

- Status: completed train + calibration; source validation passed; locked test
  inference was not run.
- Training/full-stage duration: `1205.166648100001 / 1208.3400101` seconds.
- Peak allocated CUDA memory: `1277.873046875` MiB.
- Best epoch/loss: `40 / 40`, `98305.18627929688`.
- Clipping is again active in `100%` of batches. Mean pre-clip norm averaged
  across epochs is `403047.908547852`; maximum batch norm is `2055065.375`.
- Selected top-k remains full map, `147456 / 147456` (`1.0`).
- Calibration source-group-balanced tile ROC AUC: `0.9553409090909091`.
- Calibration object/source-image ROC AUC: `0.975 / 0.88`.
- Paired DeiT-minus-ResNet primary difference is `+0.0185227272727273`,
  reversing the seed-42 difference `-0.0467045454545455`.
- Across seeds 42/123, primary means are ResNet `0.9283522727272727` and DeiT
  `0.9142613636363637`; sample SDs are `0.01197260345418126` and
  `0.05809525031794021`. Mean paired difference is `-0.0140909090909091`.
- Cross-seed calibration-score Pearson/Spearman correlations are
  `0.977389/0.973876` for ResNet and `0.920910/0.915706` for DeiT.
- Figures were visually checked against source CSV/JSON.

Interpretation: DeiT can outperform ResNet on the same frozen calibration set,
but its result is substantially more seed-sensitive. The backbone ordering is
not stable after two seeds, so the third paired seed is justified by the
predeclared protocol. Selecting seed 123 alone would be cherry-picking.

Sources:

- `fastflow_deit_base_distilled_384_printer384_v2_final/try_2_seed_123/run_note.md`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_2_seed_123/train_history.csv`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_2_seed_123/training_summary.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_2_seed_123/calibration_selection.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_2_seed_123/calibration_metrics.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_2_seed_123/calibration_report.json`

### Run 7: ResNet18-384, seed 2025

- Status: completed train + calibration; source validation passed; locked test
  inference was not run.
- Training/full-stage duration: `625.3525913999983 / 627.3981291` seconds.
- Peak allocated CUDA memory: `1954.37451171875` MiB.
- Best epoch/loss: `30 / 30`, `-2965243.3`.
- Selected top-k remains full map, `147456 / 147456` (`1.0`) for the third
  ResNet seed; the primary curve again rises toward the endpoint.
- Calibration source-group-balanced tile ROC AUC: `0.91875`.
- Calibration object/source-image ROC AUC: `0.96 / 0.92`.
- Figures were visually checked against source CSV/JSON.

Interpretation: ResNet's third seed is close to seed 42 and below seed 123;
full-map top-k is stable across all three. The paired DeiT seed 2025 remains
required before final calibration aggregation.

Sources:

- `fastflow_resnet18_384_printer384_v2_final/try_4_seed_2025/run_note.md`
- `fastflow_resnet18_384_printer384_v2_final/try_4_seed_2025/train_history.csv`
- `fastflow_resnet18_384_printer384_v2_final/try_4_seed_2025/training_summary.json`
- `fastflow_resnet18_384_printer384_v2_final/try_4_seed_2025/calibration_selection.json`
- `fastflow_resnet18_384_printer384_v2_final/try_4_seed_2025/calibration_metrics.json`
- `fastflow_resnet18_384_printer384_v2_final/try_4_seed_2025/calibration_report.json`

### Run 8: DeiT-384, seed 2025

- Status: completed train + calibration; source validation passed; locked test
  inference was not run.
- Training/full-stage duration: `1204.8147442000009 / 1207.935336` seconds.
- Peak allocated CUDA memory: `1277.873046875` MiB.
- Best epoch/loss: `40 / 40`, `81271.27047729492`.
- Clipping is active in `100%` of batches. Mean pre-clip norm averaged across
  epochs is `196183.77545`; maximum batch norm is `755359.5`.
- Selected top-k remains full map, `147456 / 147456` (`1.0`).
- Calibration source-group-balanced tile ROC AUC: `0.9426136363636364`.
- Calibration object/source-image ROC AUC: `0.97 / 0.84`.
- Paired DeiT-minus-ResNet primary difference is `+0.0238636363636364`.
- Across the three paired seeds, ResNet mean/sample SD are
  `0.9251515151515152 / 0.010120`; DeiT mean/sample SD are
  `0.9237121212121212 / 0.044221`. The mean paired difference is
  `-0.0014393939393939`, with DeiT ahead on two of three seeds.
- Figures were visually checked against source CSV/JSON.

Interpretation: the calibration results do not support a stable DeiT ranking
deficit. They support approximately equal mean ranking and materially higher
DeiT seed sensitivity. Because the comparison and full-map choice are now
predeclared from calibration, no more training or hyperparameter tuning is
justified before the locked test.

Sources:

- `fastflow_deit_base_distilled_384_printer384_v2_final/try_3_seed_2025/run_note.md`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_3_seed_2025/train_history.csv`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_3_seed_2025/training_summary.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_3_seed_2025/calibration_selection.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_3_seed_2025/calibration_metrics.json`
- `fastflow_deit_base_distilled_384_printer384_v2_final/try_3_seed_2025/calibration_report.json`

## Locked test observations

### Test 1: ResNet18-384, seed 42

- Executed once after freeze commit `b2893c7`; calibration selection was
  reused unchanged.
- Primary source-group-balanced tile ROC AUC: `0.9059381913123409` with
  per-run 95% hierarchical-bootstrap interval
  `[0.7886116094789565, 0.9854543142043142]`.
- Tile counts: `FP=5`, `FN=9`, `TN=48`, `TP=36`.
- Object-max counts: `FP=5`, `FN=2`, `TN=48`, `TP=5`.
- Source-image-max counts: `FP=2`, `FN=2`, `TN=7`, `TP=5`.
- This is an intermediate locked-test observation. Backbone interpretation is
  deferred until all three paired seeds are complete.

Sources:

- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/test_metrics.json`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/test_scores.csv`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/test_bootstrap_ci.json`
- `fastflow_resnet18_384_printer384_v2_final/try_2_seed_42/test_execution_provenance.json`

### Test 2: ResNet18-384, seed 123

- Calibration selection was reused unchanged.
- Primary source-group-balanced tile ROC AUC: `0.9327921889146379` with
  per-run 95% hierarchical-bootstrap interval
  `[0.8461595670014037, 0.9901832955404384]`.
- Tile counts: `FP=3`, `FN=11`, `TN=50`, `TP=34`.
- Object-max counts: `FP=3`, `FN=2`, `TN=50`, `TP=5`.
- Source-image-max counts: `FP=2`, `FN=2`, `TN=7`, `TP=5`.
- Versus seed 42, primary AUC changes by `+0.0268539976022970`; both
  thresholds make 14 tile errors but trade two FP for two FN.

Sources:

- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/test_metrics.json`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/test_scores.csv`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/test_bootstrap_ci.json`
- `fastflow_resnet18_384_printer384_v2_final/try_3_seed_123/test_execution_provenance.json`

## Decision log

| Date | Evidence available | Decision | Reason |
|---|---|---|---|
| 2026-07-23 | Run 1 plus frozen split audit. | Run ResNet18-384 seed 42 before DeiT. | Isolate input-resolution effect and establish the matched CNN control. |
| 2026-07-23 | User review of experiment cost. | Do not repeat ResNet18-256; use two paired seeds for the primary systems and add a third pair only if unstable. | Avoid spending nine runs solely on reproducibility. |
| 2026-07-23 | Attempt 2 failed before training; both backbones then built from the existing local cache in offline mode. | Preserve failed `try_1`; retry ResNet18-384 seed 42 as `try_2` with process-local `HF_HUB_OFFLINE=1`. | Use the same Anomalib/timm model path as the notebooks without changing proxy/VPN settings or downloading different weights. |
| 2026-07-23 | Completed ResNet18-384 seed 42: primary calibration ROC AUC `0.919886`, only `+0.005227` over ResNet18-256; both select full-map top-k. | Proceed to DeiT-384 seed 42 with the frozen MVTec-tested recipe. | Resolution alone did not materially change this seed; the matched ResNet control is valid and no extra 256 run is justified. |
| 2026-07-23 | Completed DeiT-384 seed 42: primary AUC `0.873182`, `-0.046705` versus matched ResNet; smooth loss but clipping active in `100%` of batches. | Run one DeiT-384 no-clipping ablation at seed 42 with all other settings fixed. | The baseline ranking gap is real on calibration, but saturated clipping is a measured optimizer confound. A single-factor ablation is more informative than blindly adding seeds or changing architecture. |
| 2026-07-23 | No-clip DeiT lowers normal-val loss by `20.7%`, but primary AUC changes only `+0.002159`, source AUC is unchanged, threshold accuracy is worse, and score Spearman correlation is `0.988102`. | Do not promote or repeat no-clip. Run the predeclared ResNet18-384/DeiT baseline pair at seed 123. | The clipping hypothesis is resolved without parameter fishing: clipping is not needed for stability, but it does not explain the ranking gap. A second paired seed now tests whether the gap is reproducible. |
| 2026-07-23 | ResNet18-384 seed 123 primary AUC is `0.936818`, `+0.016932` over seed 42; full-map top-k repeats. | Proceed to the already planned DeiT baseline seed 123, with no other changes. | Complete the paired comparison before interpreting backbone means or spending a third seed. |
| 2026-07-23 | DeiT seed 123 primary AUC is `0.955341`, beating paired ResNet by `+0.018523`, while seed 42 difference was `-0.046705`; DeiT two-seed SD is `0.058095`. | Run the predeclared third pair at seed 2025, ResNet first and DeiT second. | Ordering is inconsistent and DeiT variance is material. A third paired seed is required to avoid cherry-picking either outcome. |
| 2026-07-23 | ResNet18-384 seed 2025 primary AUC is `0.91875`; full-map top-k repeats for all three ResNet seeds. | Proceed to paired DeiT seed 2025 with no config changes. | Complete the already justified third pair before aggregation and freeze. |
| 2026-07-23 | DeiT seed 2025 primary AUC is `0.942614`, beating paired ResNet by `+0.023864`; three-seed mean difference is `-0.001439`, with DeiT SD `0.044221` versus ResNet `0.010120`. All six primary runs select full-map top-k. | Stop training, freeze both primary recipes and calibration selections, then perform one locked-test evaluation for all three paired seeds. | Calibration means are effectively tied and another variant would increase tuning risk. The remaining question is whether DeiT's higher seed sensitivity transfers to the untouched test. |

## Update checklist after each attempt

1. Verify metric/report values against source CSV/JSON and hashes.
2. Update the run's `run_note.md` with outcome and decision.
3. Update `printer384_v2_experiment_ledger.csv`, including failed runs.
4. Update hypothesis status here as `supported`, `not supported`, or
   `inconclusive`; retain contradictory evidence.
5. Commit compact artifacts and this register before starting another run.
