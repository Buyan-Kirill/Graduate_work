# Project Handoff Memory

Last updated: 2026-07-23

This file is meant to be pasted into a new chat so work can continue without rebuilding context.

## Priority Runbook: FastFlow Printer Backbone Study

### Approval state

- Autonomous work was approved on 2026-07-23 after reducing the seed matrix
  and adding explicit anti-overfitting, provenance, and notebook requirements.

### Objective and honest acceptance criteria

Determine whether FastFlow with `deit_base_distilled_384` can match or exceed
the ResNet18 systems on the frozen printer dataset. If it cannot, identify the
most likely cause using saved measurements and controlled ablations rather
than speculation.

- Primary ranking metric: source-image-balanced tile ROC AUC.
- Always report raw DeiT-minus-ResNet ROC AUC and paired 95% cluster-bootstrap
  interval. Do not assume an arbitrary ROC AUC non-inferiority margin.
- Practical threshold criterion: for every paired seed, DeiT may make at most
  one additional misclassified test tile versus the relevant ResNet baseline.
- Always report tile FP/FN/error counts and source-image-max FP/FN/error counts.
- Ideally DeiT should have a non-negative raw difference and fewer errors.
- Because test has only 9 normal and 7 anomalous source groups, conclusions
  must explicitly retain statistical and domain-generalization limitations.

### Frozen data and current state

- Dataset split: `printer_384_v2_final`.
- Manifest:
  `experiments/printer/dataset_v384_audit/printer_split_v2_final.csv`.
- Manifest SHA-256:
  `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
- Never move, delete, or overwrite original dataset files. Review folders are
  copies and are not model inputs.
- The locked test has not been read by the new protocol.
- Completed training count: 8. One additional attempt failed before training
  because network metadata access was unavailable and is preserved in ledger.
- Completed calibration controls: `resnet18_256`, seed 42 has primary ROC AUC
  `0.914659090909091`; `resnet18_384`, seed 42 has `0.9198863636363637`.
  Both selected the full anomaly map as top-k.
- Completed DeiT-384 seed 42 calibration has primary ROC AUC
  `0.8731818181818182` and also selects the full map. Its clipping threshold
  `10.0` was exceeded in every batch. A completed no-clipping ablation lowers
  normal-val loss by `20.7%` but changes primary AUC by only `+0.002159` and
  worsens threshold balanced accuracy, so clipping does not explain the
  ResNet ranking advantage. Do not repeat no-clip at seed 123.
- The three paired calibration seeds are complete. ResNet18-384 mean/sample SD
  are `0.9251515151515152 / 0.010120`; DeiT mean/sample SD are
  `0.9237121212121212 / 0.044221`. Paired DeiT-minus-ResNet differences are
  `-0.046705`, `+0.018523`, and `+0.023864`, with mean `-0.001439`.
  All six primary runs selected full-map top-k. Stop training and freeze the
  two recipes plus per-run calibration thresholds before one locked-test pass
  over the six primary checkpoints. Test remains locked at this handoff.
- Current execution/protocol base commit: `f848062`.

### Non-negotiable execution limits

- Run exactly one training process at a time. Never parallelize GPU training.
- Check that no prior Python/training process is active and inspect
  `nvidia-smi` before every run.
- Normal run duration should be at most about 40 minutes. Investigate a run
  that materially exceeds this before starting another one.
- Expected core budget: 5 runs including the completed run; at most 7 when a
  third paired seed is needed. Allow at most 2 evidence-based follow-up runs,
  for an expected total no larger than 9.
- The earlier 10-12 run allowance is a ceiling, not a target.
- Hard stop: 15 training runs or 16 hours of autonomous work, whichever comes
  first. At that point, stop and write conclusions from available evidence.
- Do not perform broad hyperparameter search. Every follow-up must test a
  specific hypothesis supported by prior artifacts.
- Never use `--allow-overwrite` or `--skip-hash-verification` for real runs.
- Preserve failed and unexpectedly poor runs. Never edit metrics to fit the
  expected conclusion.

### Preflight before every training run

1. Confirm the latest user instruction still permits autonomous execution.
2. Confirm Git working state and current commit. Prefer a clean committed tree;
   otherwise record the exact dirty-state and file hashes in provenance.
3. Confirm `.git/objects/maintenance.lock` is absent. If Git maintenance is
   active but consuming CPU, let it finish; do not kill it without permission.
4. Confirm no other training process is active and the GPU has enough free
   memory.
5. Verify the split config and manifest hash. Do not proceed on a mismatch.
6. Confirm the destination run directory does not contain prior artifacts.
7. Add a planned ledger entry/run note with run number, hypothesis, config,
   seed, and reason for the run.

### Baseline experiment sequence

Use separate CLI invocations for one config and one seed so execution is
unambiguously sequential.

Phase A, resolution/backbone comparison on seed 42:

1. Run 2: `resnet18_384`, seed 42.
2. Run 3: `deit_base_distilled_384`, seed 42.
3. Validate each run report and compare calibration-only evidence: loss curve,
   best epoch, top-k sweep, score distribution, gradient clipping activity,
   and tile/object/source ranking metrics.
4. Stop for a decision gate. Do not launch all remaining seeds blindly if the
   first DeiT run is invalid, unstable, or exposes an implementation issue.

Phase B, paired reproducibility after Phase A passes validation:

5. Run `resnet18_384`, seed 123.
6. Run `deit_base_distilled_384`, seed 123.
7. Compare both paired seeds. Add seed 2025 for both primary systems only if
   the ordering is inconsistent, variability is material, or one run is
   technically invalid.

The historical `resnet18_256`, seed 42 run remains a secondary reference and
is not repeated. The normal core therefore contains 5 total runs: that
historical run plus two paired seeds for ResNet18-384 and DeiT-384. The maximum
core contains 7 total runs when the third paired seed is justified.

### Calibration-only decision gate

After every run, and again after the paired baseline matrix:

- Verify all report values against `train_history.csv`,
  `training_summary.json`, `calibration_selection.json`,
  `calibration_metrics.json`, `calibration_scores.csv`, and the top-k sweep.
- Inspect seed variance and whether top-k choices are stable or are driven by
  the small set of 5 normal and 5 anomalous calibration source groups.
- Distinguish optimization failure from representation failure:
  - unstable/non-converged loss, clipping saturation, or endpoint best epoch
    supports an optimizer/schedule hypothesis;
  - stable likelihood training but consistently weak anomaly ranking supports
    a feature-representation or spatial-resolution hypothesis;
  - large changes across seeds/top-k support calibration variance rather than
    a stable backbone conclusion.
- Relevant architecture fact to test against observations: the current
  Anomalib ResNet18 FastFlow path uses three feature scales, while the DeiT
  path exposes one 24x24 patch-token feature map at 384 input. This is a
  plausible disadvantage for small local printer defects, but it is not a
  conclusion until results support it.

### Targeted follow-ups

Reserve at most 2 training runs after the paired baseline matrix. They may
evaluate at most one alternative DeiT recipe, first on seed 42 and then on seed
123 only if the first result supports the predeclared hypothesis. Select the
alternative only from calibration evidence and document it before execution.

- Optimization follow-up only if DeiT training diagnostics show instability
  or under-training: change one of LR, schedule/epochs, or clipping behavior,
  not several at once.
- Post-processing follow-up only if calibration top-k curves show a stable
  cross-seed region. Never choose top-k from test.
- Representation follow-up only if optimization is healthy but DeiT ranking
  remains weak: test one explicit feature-resolution/multi-layer hypothesis.
  Treat it as a new model configuration with a new run directory.
- If calibration evidence is too noisy to choose among alternatives, do not
  spend runs on parameter fishing; retain the baseline and report uncertainty.

To limit adaptive overfitting to the small calibration split, do not test a
second alternative recipe, do not change multiple factors in one run, and do
not choose a single lucky seed. Choose a final recipe from aggregate paired-seed
calibration evidence plus top-k/threshold stability, then freeze it.

### Freeze and locked test

1. Freeze all compared configs, checkpoints, top-k values, thresholds, seeds,
   and analysis code in a documented local commit before test.
2. Push that commit by explicit HTTPS URL.
3. Run the `test` stage once for every frozen run, sequentially. The test stage
   must load `calibration_selection.json` and must not select top-k/threshold.
   Here "locked" means no test image decoding, inference, score inspection, or
   metric computation before freeze; manifest metadata and integrity hashes
   may be accessed by validation code.
4. Run `code/analyze_fastflow_printer_results.py` without an implicit ROC AUC
   margin. Save raw differences, paired intervals, and exact threshold errors.
5. Test results may be used for final error analysis, but not for another
   tuning loop presented against the same test as unbiased. Any post-test model
   change requires a new future holdout for an honest confirmatory claim.

### Required artifacts and human observability

Every attempted run, successful or failed, must have a unique directory and a
ledger row. Preserve at least:

- `run_note.md`: hypothesis, exact change, expected diagnostic, outcome, and
  next decision;
- `run_config.json` and `execution_provenance.json`: seed, config, Git state,
  hashes, environment, and timestamps;
- `train_history.csv`, `training_summary.json`, and checkpoints/weights;
- calibration selection, metrics, scores, top-k sweep, and validated report;
- loss/LR, top-k, and score-distribution plots;
- after freeze only: test scores/metrics and final comparison reports.

Update `experiments/printer/printer384_v2_experiment_ledger.csv` after every
attempt, including failures and elapsed time. Before citing a number, trace it
to the source artifact and verify report consistency/hashes.

The central evidence memory is
`experiments/printer/fastflow_printer384_v2_findings.md`. After every attempted
run and before another training starts, update its hypothesis status, exact
verified evidence, artifact links, contradictions, and next decision. Keep the
individual `run_note.md` as the detailed record for that run; do not rely on
chat memory alone.

Before run 2, keep provenance lightweight but sufficient:

- require a clean Git commit at training start so the commit identifies all
  source code; also save hashes of the pipeline and CLI entry point;
- save only relevant software versions, Python/PyTorch/CUDA information, GPU
  model, deterministic mode, and peak allocated CUDA memory;
- retain per-epoch loss/LR/duration and DeiT gradient norm/clipping summaries;
- do not save batch-level logs, activations, full anomaly-map tensors, embedded
  notebook images, or a full `pip freeze` unless needed for diagnosis;
- keep weights and resumable checkpoints locally under the existing `*.pth`
  ignore rule; commit compact CSV/JSON/Markdown reports and selected PNG plots.

### Final reproducibility notebook

After the final ResNet18 and DeiT recipes are frozen, create one concise
`code/FastFlow_printer_final_reproduction.ipynb` that:

- pins/displays the manifest hash, Git commit, exact config, and chosen seed;
- calls the shared pipeline rather than duplicating training implementation;
- defaults to a new run directory and refuses overwrite;
- separates `train_calibrate`, locked `test`, and `load/report` modes;
- displays training curves, top-k calibration behavior, score distributions,
  final metrics, and representative error examples after test;
- reproduces the final selected recipes, not a seed cherry-picked by test;
- is committed with outputs cleared so Git stores the executable notebook code
  without heavy embedded artifacts. Generated results remain in run folders.

### Git and versioning rules

- Never rewrite history, delete old experiment data, or use destructive Git
  commands.
- Commit code/protocol changes before the run they affect. Commit run metadata,
  metrics, plots, reports, and ledger updates after each run or one small,
  clearly identified batch. Weights remain ignored if already covered by
  `.gitignore`.
- The configured `origin` may remain SSH. For network writes use exactly:
  `git push https://github.com/Buyan-Kirill/Graduate_work.git main`.
- Do not change `git config`, remotes, credentials, proxy, VPN, SSH, or machine
  settings. If HTTPS push fails, make a local commit, record that it is not
  pushed, stop repeated attempts, and continue only when local versioning is
  safe.
- Never push while Git maintenance holds its lock. Wait and verify CPU activity
  before deciding that maintenance is stuck.

### Communication and final stop conditions

- Notify the user at important boundaries: preflight, start of each training
  run, completion with exact calibration facts, decision gate, config freeze,
  test start, and final conclusion. Do not stream noisy batch progress.
- Ask the user only for domain-label decisions, physical machine intervention,
  or choices outside this approved runbook. State exactly what is needed and
  why.
- Stop early for invalid data/provenance, repeated execution failure, hardware
  risk, evidence that the planned comparison is methodologically invalid, or
  the run/time hard limit.
- Final report must list every run, hypothesis, config/seed, duration, exact
  calibration/test metrics, error counts, relevant plots/artifact paths, and
  Git commits. Clearly separate facts, supported interpretations, unresolved
  uncertainty, and production limitations.

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

## FastFlow Printer Update (2026-07-22)

- Current dataset: `datasets/processed_printer_dataset_384`.
- Approved leak-free manifest: `experiments/printer/dataset_v384_audit/printer_split_v2_final.csv`.
- Split version: `printer_384_v2_final`; train/normal validation/calibration/test are disjoint by source capture, object group, and SHA-256.
- Domain review is complete. One out-of-scope anomaly and two ambiguous normal object groups are excluded; accepted normal deviations remain as intentional hard negatives.
- Current notebooks: `code/FastFlow_printer_resnet18.ipynb` and `code/FastFlow_printer_deit.ipynb`.
- Shared implementation: `code/fastflow_printer_pipeline.py`.
- Staged CLI: `code/run_fastflow_printer_experiments.py`; use `train-calibrate` first and `test` only after configurations are frozen.
- Calibration diagnostics: `code/analyze_fastflow_printer_calibration.py`.
- Run three paired seeds: 42, 123, 2025. Select top-k only on calibration and never tune from test.
- After locked test, run `code/analyze_fastflow_printer_results.py` for paired hierarchical-bootstrap comparison.
- Full protocol and limitations: `experiments/printer/fastflow_printer384_v2_protocol.md`.

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
