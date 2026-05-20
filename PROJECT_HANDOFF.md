# Project Handoff Memory

Last updated: 2026-05-20

This file is meant to be pasted into a new chat so work can continue without rebuilding context.

## Short Memory

Project goal: anomaly detection and localization for metal 3D-printer parts. MVTec AD is used as a controlled benchmark/sanity lab, especially `metal_nut` and `hazelnut`, but final confidence must come from the real 3D-printer dataset.

Current main notebook work:

- `code/SuperSimpleNet_mvtec.ipynb` is the actively refactored MVTec SuperSimpleNet notebook.
- `code/SuperSimpleNet_mvtec_unsupervised_compare.ipynb` was added for Experiment C on `metal_nut`: compare synthetic unsupervised proxy metrics with real labeled reference metrics.
- It now uses explicit SSN inference via `predict_anomaly_map`, no pseudo-TTA.
- It uses a normal `train/val` split for synthetic `Validation_Loss`.
- Labeled segmentation/classification data, when available, is split separately into validation/test metric sets.
- If labeled segmentation/classification is missing, final evaluation can fall back to synthetic metrics from held-out normal images. These metrics must be named `synthetic_*` and treated as diagnostics, not real benchmark metrics.
- Early stopping was changed by the user to `Validation_Loss` with mode `min`. This is accepted as a good default for the weak/unsupervised target setting.
- Epoch logs should stay compact: `Train_Loss`, `Validation_Loss`, `pixel_aupro`, `image_roc_auc_raw`.
- Per-class metrics are intended for final reporting only, not every epoch.
- Intermediate validation visualizations should save one example per anomaly class at `middle_logs/visualizations/epoch_<N>/`.
- Final test visualizations save one example per anomaly class at `test_visualizations/`.

Important files updated in this chat:

- `code/SuperSimpleNet_mvtec.ipynb`
- `code/SuperSimpleNet_mvtec_unsupervised_compare.ipynb`
- `code/SuperSimpleNet_printer.ipynb`
- `code/SuperSimpleNet_printer_head_finetune.ipynb`
- `plans.md`
- `experiments/README.md`
- `experiments/metal_nut/README.md`

Latest addition:

- New notebook: `code/SuperSimpleNet_mvtec_unsupervised_compare.ipynb`.
- Output folder: `experiments/metal_nut/ssn_unsupervised_proxy_compare/try_1`.
- Dataset setup: one normal train subset; validation and test each have fixed synthetic segmentation/classification references plus real segmentation/classification references.
- Synthetic references are generated once before training as fixed SSN synthetic masks/labels. During validation/test those fixed masks are reused through the current model path (`feature_extractor -> adaptor -> anomaly_generator -> segdec`), following the training-path notes from `code/ssn_calculator.ipynb`.
- Early stopping remains unsupervised on `Validation_Loss`; in this notebook it uses the fixed synthetic validation reference. Real labels/masks are used only for reference comparison and final per-defect reporting.
- Final outputs include `test_metrics.txt`, `test_metric_comparison.csv`, real/synthetic validation visualizations, and real/synthetic test visualizations.

Known current git status at time of this handoff: `experiments/metal_nut/README.md` is modified.

## Long Memory

### Repository Shape

Root path in this session:

`C:\Users\AERO\Desktop\master_work\Experiments`

Important folders:

- `code/`: notebooks and data preparation scripts.
- `experiments/`: metrics, plots, model weights/logs from runs.
- `experiments/metal_nut/`: MVTec `metal_nut` runs.
- `experiments/hazelnut/`: MVTec `hazelnut` runs and weak-supervised planning.
- `datasets/`: local datasets, generally not for GitHub.

Main SSN notebooks:

- `code/SuperSimpleNet_mvtec.ipynb`
- `code/SuperSimpleNet_printer.ipynb`
- `code/SuperSimpleNet_printer_head_finetune.ipynb`
- `code/ssn_calculator.ipynb` was used as a reference for correct SSN forward/inference behavior.

### Key Technical Decisions

#### Correct SSN inference path

`output.anomaly_map` from `SupersimplenetModel.forward()` is raw logits but already upsampled through `model.anomaly_map_generator(...)` to the input size. Do not manually replace that with a plain `torch.nn.functional.interpolate` when reproducing the model forward.

Correct probability map for visualization/pixel metrics:

```python
anomaly_prob = torch.sigmoid(output.anomaly_map)
```

Image score:

- `output.pred_score` is raw image-level logit.
- ROC AUC can use raw logits because sigmoid is monotonic.
- If sigmoid score is saved/displayed, call it `sigmoid_pred_score`, not just `image_score`.

#### No pseudo-TTA

Old misleading `predict_with_augmentations`/flip-TTA was removed from active code. New code should call:

```python
predict_anomaly_map(model, img, true_img_size=None, apply_sigmoid=True)
```

TTA should only be reintroduced as an explicit `predict_with_tta` after validation, with correct inverse transforms and metric comparison.

#### Training loss path

Training and validation loss should use one helper:

```python
ssn_training_forward(model, images, masks=None, labels=None)
```

This runs:

`feature_extractor -> adaptor -> downsample_mask -> anomaly_generator -> segdec`

Important: synthetic anomaly diagnostics/loss validation must run with `model.train()` because `model.anomaly_generator` can depend on train/eval mode. Use `torch.no_grad()` for validation loss to avoid gradients, but keep train mode for synthetic generation.

#### Validation/test split logic for MVTec

Preferred current scheme in `SuperSimpleNet_mvtec.ipynb`:

1. Split MVTec `train/good` into:
   - `train_dataset`
   - `normal_validation_dataset`

2. `normal_validation_dataset` is used for synthetic `Validation_Loss`.

3. If real segmentation labels/masks exist:
   - create `segmentation_dataset`;
   - split by index into `segmentation_val_dataset` / `segmentation_test_dataset`;
   - use validation split for monitoring aggregate `pixel_aupro`;
   - use test split for final aggregate and per-class pixel metrics.

4. If real classification labels exist:
   - create `classification_dataset`;
   - split by index into `classification_val_dataset` / `classification_test_dataset`;
   - use validation split for monitoring aggregate `image_roc_auc_raw`;
   - use test split for final aggregate and per-class image metrics.

5. If a labeled task is unavailable:
   - use synthetic fallback from held-out normal images;
   - name metrics as `synthetic_*`;
   - treat them as diagnostics/regression checks, not real benchmark scores.

#### Early stopping

The user changed early stopping to `Validation_Loss`. This is accepted.

Rationale:

- `Validation_Loss` on held-out normal images with synthetic anomalies checks both segmentation and image score components through `SSNLoss`.
- It better matches the future weak/unsupervised setting where real masks may be unavailable.
- Real `pixel_aupro` and `image_roc_auc_raw` should still be logged as monitoring metrics when available.

Default should be:

```python
early_stopping_metric="Validation_Loss"
early_stopping_mode="min"
```

### Per-Class Metrics and Visualizations

The user asked for per-anomaly-class validation, but clarified it should not spam every epoch.

Current desired behavior:

- Epoch logs: only aggregate metrics.
- Final test report: aggregate plus per-class metrics.
- Intermediate visualizations: one saved image per anomaly class at `log_interval` epochs, but not per-class scalar spam in logs.

Per-class metric names:

- `pixel_aupro_<defect_type>`
- `image_roc_auc_raw_<defect_type>`

For image-level per-class ROC AUC, compare `good` vs each anomaly class.

### Visualizations

Visualization requirements:

- Use `vmin=0.0, vmax=1.0` for anomaly heatmaps.
- Denormalize input images and clamp to `[0, 1]`.
- Do not show raw logits as probability maps.
- Include foreground diagnostic/masked overlay as visual aid only; it should not replace full-image benchmark metrics.

Added/desired paths:

- Validation visualizations: `middle_logs/visualizations/epoch_<N>/`
- Final test visualizations: `test_visualizations/`

### Synthetic Dataset / Proxy Metric Idea

The user is interested in weak-supervised mode where real anomaly masks may not exist. In that setting, real segmentation can only be judged visually.

Idea recorded in `plans.md` as `Experiment C: synthetic segmentation metric as a proxy`:

- Build synthetic segmentation evaluation from held-out normal images and generated masks.
- Compare synthetic segmentation metrics to real mask metrics on datasets where real masks exist.
- Track both across epochs/checkpoints.
- Measure correlation/rank agreement.
- Decide whether synthetic metrics are reliable enough for weak-supervised early stopping/regression monitoring.

Important naming:

- `synthetic_pixel_aupro`
- `synthetic_pixel_roc_auc`
- `synthetic_pixel_ap`

Do not confuse these with real MVTec-style pixel metrics.

### MVTec Class Relevance Discussion

For final target: metal 3D-printer part defect detection/localization.

MVTec AD usage:

- Good benchmark/sanity lab.
- Not sufficient proof of production/domain performance.

`metal_nut`:

- Closest MVTec class among discussed categories.
- Useful because it is a metal object with shape, hole, reflections, scratches/bent/color/flip defects.
- Still not fully representative of 3D-printed metal defects such as roughness, powder residue, pores, layer artifacts, lack-of-fusion, balling, support marks, geometric build defects.

`hazelnut`:

- Domain material is far from target.
- Still useful as localization stress test for cracks/cuts/holes/irregular surfaces.
- Good for weak-supervised ablations because masks can be hidden during training and used only for evaluation.

Conclusion:

- Use `metal_nut` as primary MVTec metal-like sanity benchmark.
- Use `hazelnut` as auxiliary localization stress test.
- Ultimately collect/label a small real printer-part test set, ideally with some masks.

### Plans and Experiment Notes

`plans.md` now contains:

- Weak/mixed SuperSimpleNet experiment plan for MVTec hazelnut.
- Experiment A: when to add real anomalies.
- Experiment B: defect-type generalization.
- Experiment C: synthetic segmentation metric as a proxy.
- Suggested next important implementation: mixed second stage with synthetic SSNLoss preserved while adding weak real anomalies.

Important current interpretation from previous hazelnut results:

- Baseline localization already strong before head fine-tune.
- Classification head fine-tune improved image AUROC without changing segmentation maps.
- Weak segmentation decoder fine-tune using image-level losses degraded localization.
- Supervised mask fine-tune improved pixel AP strongly.
- Therefore, weak image-level gradients can help classification but may damage localization unless synthetic/masked supervision is preserved.

### README Updates

`experiments/README.md` now has `Current SSN evaluation logic`:

- normal `train/val` split for synthetic validation loss;
- labeled segmentation/classification separate val/test splits;
- synthetic fallback named `synthetic_*`;
- final per-defect metrics and one-per-class visualizations;
- compact epoch logs.

`experiments/metal_nut/README.md` has `try_6`:

- planned run with updated `code/SuperSimpleNet_mvtec.ipynb`;
- normal train/val for synthetic val loss;
- labeled val/test metrics;
- synthetic fallback for missing labels;
- final per-class metrics and one-per-class visualizations.

Note: several README files display mojibake in PowerShell. Avoid rewriting existing non-ASCII text unless necessary. Prefer appending short ASCII sections to avoid encoding churn.

### Known Recent Bug Fixes

`save_validation_visualizations(..., device=device)` caused:

```text
NameError: name 'device' is not defined
```

because default arguments are evaluated at function definition time. It was fixed to:

```python
def save_validation_visualizations(model, segmentation_dataset, output_dir, device=None):
    if device is None:
        device = next(model.parameters()).device
```

### Verification Already Done

During the refactor, code cells in these notebooks were repeatedly checked with `ast.parse`:

- `code/SuperSimpleNet_mvtec.ipynb`
- `code/SuperSimpleNet_printer.ipynb`
- `code/SuperSimpleNet_printer_head_finetune.ipynb`

Full notebook execution/training was not run because it requires CUDA kernel/environment and would be long.

### Useful Next Tasks

Likely next work:

1. Run `code/SuperSimpleNet_mvtec.ipynb` for `metal_nut` `try_6`.
2. Inspect:
   - `final_metrics.txt`
   - `test_metrics.txt`
   - `middle_logs/checkpoints_metrics.txt`
   - `middle_logs/visualizations/epoch_<N>/`
   - `test_visualizations/`
3. Confirm epoch logs are compact and final test has per-class metrics.
4. Compare early stopping by `Validation_Loss` against best real `pixel_aupro`/`image_roc_auc_raw` checkpoints if checkpoint history is available.
5. Add per-defect summary CSV/JSON if final text metrics become inconvenient.
6. For weak-supervised experiments, implement mixed fine-tune stage:
   - real anomalies: image-level loss;
   - clean normal images: negative image/map regularization;
   - synthetic anomalies: full SSNLoss with masks.

### Style / Process Preferences

User prefers:

- careful notebook refactoring;
- minimal unrelated edits;
- not adding noisy logs;
- concise README updates;
- honest ML-engineering reasoning, especially about metric validity and domain shift.
