# FastFlow printer384 v2 protocol

## Status

The dataset split is frozen. The experiment is staged so that training and
calibration can be analyzed before the locked test is read. Full GPU training
has not yet been run for this protocol.

Goal: compare a DeiT-backed FastFlow system against ResNet18 without selecting
post-processing on test. The result may support practical non-inferiority on
this internal dataset, but it cannot by itself establish production
generalization to future print sessions.

## Data

- Dataset root: `datasets/processed_printer_dataset_384`.
- Split version: `printer_384_v2_final`.
- Manifest: `dataset_v384_audit/printer_split_v2_final.csv`.
- Manifest SHA-256:
  `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
- Split config: `configs/printer_split_v2_final.json`.

| Split | Good tiles | Anomaly tiles | Good source images | Anomaly captures |
|---|---:|---:|---:|---:|
| train | 1000 | 0 | 212 | 0 |
| normal_val_loss | 160 | 0 | 44 | 0 |
| calibration | 40 | 21 | 5 | 5 |
| test | 53 | 45 | 9 | 7 |

Normal splits are disjoint by original source image and object group. Anomaly
calibration/test splits are disjoint by source capture. Every pair of splits
is also disjoint by exact SHA-256.

Domain review decisions are in
`dataset_v384_audit/printer_split_v2_final_domain_review.md`. Accepted normal
images with permissible deviations remain in `good` as hard negatives. The
visual `_review_split_*` folders are copies only and are not training inputs.

## Systems

| Config | Input | Epochs | Flow steps | LR | Weight decay | Batch | Clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18_256 | 256 | 14 | 10 | 1e-3 | 1e-3 | 8 | none |
| resnet18_384 | 384 | 30 | 10 | 1e-3 | 1e-3 | 8 | none |
| deit_base_distilled_384 | 384 | 40 | 8 | 3e-5 | 1e-5 | 10 | 10.0 |

`resnet18_256` preserves the historical printer baseline.
`resnet18_384` is the primary resolution-matched control. DeiT uses the
architecture-specific recipe that succeeded in the MVTec comparison rather
than copying ResNet optimizer settings. No train augmentation is enabled in
this comparison. Gradient clipping activity for DeiT is logged per epoch.

Each system is trained with seeds 42, 123, and 2025. The pretrained feature
extractor is frozen. The best checkpoint is selected only by likelihood loss
on disjoint normal data in `normal_val_loss`.

## Calibration and locked test

For each trained model, the `train-calibrate` stage:

1. Sweep the predefined top-k fraction grid on labeled `calibration` only.
2. Select by source-image-balanced tile ROC AUC, with object/tile ROC AUC, AP,
   and smaller k as deterministic tie-breakers.
3. Set the operating threshold to the fixed 95th percentile of normal
   calibration tile scores.
4. Save the selected top-k and threshold to `calibration_selection.json`
   without reading test.

After all candidate configurations are frozen, the separate `test` stage
loads that file and evaluates test once. It cannot select a new top-k.

The same tile threshold is also applied after object/source max aggregation to
show the operating false-positive cost of an "any anomalous tile" rule. Those
aggregated threshold metrics are not separately calibrated thresholds.

## Reporting

Primary metric: source-image-balanced tile ROC AUC. Secondary metrics:
unweighted tile ROC AUC/AP, max-aggregated object ROC AUC/AP, max-aggregated
source-image ROC AUC/AP, and threshold balanced accuracy/FPR/FNR.

Per-run 95% intervals use label-stratified cluster bootstrap over source
images. The final DeiT-minus-ResNet comparison uses a paired hierarchical
bootstrap over both training seeds and source images. The raw ROC AUC
difference and its confidence interval are always reported. No ROC AUC
non-inferiority margin is assumed by default; the script produces that binary
decision only when a margin is explicitly supplied.

For the thresholded result, the practical criterion is no more than one
additional misclassified test tile for DeiT versus a ResNet baseline for every
seed. False positives, false negatives and total errors are also reported
after max aggregation by source image. This source-level view is required
because tiles from one capture are correlated; the one-tile tolerance is not a
claim that all 98 test tiles are statistically independent.

Important limitations:

- calibration contains only 5 normal and 5 anomalous source groups;
- test contains only 9 normal and 7 anomalous source groups;
- multiple tiles from one capture are correlated and do not increase the
  number of independent test units;
- three training seeds give only a coarse estimate of optimization variance;
- no pixel-level defect masks exist, so localization quality is not measured;
- a future-session holdout is still required for an external production claim.

## Execution budget and observability

- Only one GPU training process may run at a time.
- Target budget: 10-12 training runs.
- Hard stop: 15 training runs or 16 hours of total autonomous work.
- Every run must have a separate directory, `run_note.md`, `run_config.json`,
  epoch history, calibration artifacts and a validated calibration report.
- Training and locked test require a clean Git worktree. The run records the
  commit, pipeline/CLI hashes, relevant package and CUDA/GPU versions,
  deterministic mode, and peak allocated CUDA memory.
- The central run list is `printer384_v2_experiment_ledger.csv`.
- `code/report_fastflow_printer_calibration.py` verifies source consistency and
  creates loss/LR, top-k and score-distribution plots without modifying the
  source metrics.
- Failed or worse-than-expected runs remain in the ledger and final report.
- No value may be copied into a summary unless it can be traced to a saved
  training/evaluation artifact.
- Model weights and resumable checkpoints remain local under the `*.pth`
  ignore rule. Compact metrics, provenance, Markdown reports and selected PNG
  plots are versioned; batch logs, activations and full anomaly maps are not.
- Models are created through Anomalib `FastflowModel(pre_trained=True)`, as in
  the printer notebooks. When network metadata access is unavailable but the
  exact timm weights are already cached, runs use process-local
  `HF_HUB_OFFLINE=1` and record that value in provenance; proxy/VPN settings
  are not changed.

## Run order

1. `.venv\Scripts\python.exe code\run_fastflow_printer_experiments.py --stage train-calibrate`
2. `.venv\Scripts\python.exe code\analyze_fastflow_printer_calibration.py`
3. Freeze the configurations after calibration-only analysis.
4. `.venv\Scripts\python.exe code\run_fastflow_printer_experiments.py --stage test`
5. `.venv\Scripts\python.exe code\analyze_fastflow_printer_results.py`

The two notebooks expose the same `train_calibrate`, `test`, and `load` modes
for interactive use.

All new runs use `*_printer384_v2_final/try_1_seed_*` directories and refuse
to overwrite existing artifacts by default. Historical `fastflow_resnet18`
and `fastflow_deit_base_distilled` directories remain untouched.
