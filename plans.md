# Plans: weak/mixed SuperSimpleNet experiments on MVTec hazelnut

## Context

Current experiment: `experiments/hazelnut/ssn_head_finetune`.

Important current result:

- `before_head_finetune`: strong localization already: pixel AUPRO ~0.9318, pixel ROC AUC ~0.9803, pixel AP ~0.5096.
- `after_classification_head_finetune`: image AUROC improves to ~0.9959 and pixel metrics stay unchanged, because the segmentation map is not updated.
- `after_weak_segdec_classification_loss_finetune`: pixel metrics degrade noticeably: AUPRO -0.0605, pixel ROC -0.0274, pixel AP -0.0563.
- `after_weak_segdec_topk_mil_finetune`: image AUROC reaches 1.0, but localization still degrades: AUPRO -0.0182, pixel AP -0.0337.
- `after_segdec_mask_finetune`: supervised masks strongly improve pixel AP: +0.2676, pixel ROC +0.0108, AUPRO roughly unchanged.

Interpretation: weak image-level gradients can help classification, but currently they are too noisy/destructive for the segmentation map. The next experiments should preserve synthetic/masked localization supervision while adding real anomalies carefully.

## General rules

Use fixed seeds and fixed train/fine-tune/holdout splits across all runs. Compare runs by deltas against the same baseline.

Primary metrics:

- `image_roc_auc_raw` for image-level anomaly detection;
- `pixel_aupro` for localization/segmentation.

Secondary diagnostics, only if a specific question needs them:

- image/pixel AP for severe class imbalance or thresholding analysis;
- per-defect-type breakdowns for `crack`, `cut`, `hole`, `print`.

Always save:

- `comparison_metrics.json` and `.csv`;
- per-image scores;
- per-defect summary;
- a few visualizations per defect type;
- config with exact anomaly fractions, defect types used, seed, LR, epochs, loss weights.

If a new notebook or new experiment folder is created, update the nearest existing `README.md`.


## Logging convention

Use one root folder per experiment attempt, for example `experiments/hazelnut/ssn_head_finetune/try_3`. The baseline notebook follows the same pattern: `experiments/hazelnut/ssn_baseline/<try_N>`.

Inside that folder, keep one subfolder per evaluated model state. Do not create separate train/eval folders for the same state. For the current notebook the expected run folders are:

- `before_head_finetune`;
- `after_classification_head_finetune`;
- `after_weak_segdec_classification_loss_finetune`;
- `after_weak_segdec_mixed_conservative_finetune`;
- `after_segdec_mask_finetune` if the supervised-mask upper bound is enabled.

Each run folder should contain its own training artifacts when applicable, evaluation metrics, per-image scores and visualizations. Root-level `comparison_metrics.json` / `.csv` compare these run folders.

## Experiment A: when to add real anomalies

Goal: compare whether real anomalies should be used from the start, only during fine-tune, or gradually mixed in.

Use the same limited number of real anomaly images in all variants. Keep holdout images fixed and never use them for training/fine-tuning.

Recommended variants:

1. `A0_current_two_stage`

Current baseline: train SSN on normal images with synthetic feature anomalies, then fine-tune heads on a limited real-anomaly set.

2. `A1_joint_mixed_from_start`

Train in one stage with a mixed loader:

- normal images with synthetic feature anomalies and masks;
- real anomalous images with image-level labels;
- clean normal images as negative samples.

If real masks are available, run a supervised/mixed version too. If simulating weak supervision, hide masks for the real anomalous branch and use them only for final evaluation.

3. `A2_curriculum_real_anomalies`

Start like normal SSN training. Then gradually increase the probability of real anomalies, for example:

- first 50-60% epochs: only normal + synthetic anomalies;
- next 20-30% epochs: 10-25% real anomalies in batches;
- final epochs: target real anomaly ratio, for example 25-50% depending on dataset size.

This tests whether the segmentation map benefits from first learning generic synthetic localization before seeing scarce real defects.

4. `A3_two_stage_mixed_finetune_with_synthetic`

Most important next variant. Keep current base training, but replace weak fine-tune with a mixed second stage:

- real anomalous images without masks: classification loss, optionally weak map loss;
- clean normal images: negative image loss and light map false-positive suppression;
- synthetic anomalies generated from normal images: full SSNLoss with synthetic masks.

This is the lowest-risk extension of the current notebook because `model.anomaly_generator` and `SSNLoss` are already used.

Success criterion for A: real anomalies should improve image metrics without reducing pixel AUPRO/AP versus `before_head_finetune`. A variant that improves image AUROC but hurts pixel AP is not a win for segmentation.

## Experiment B: defect-type generalization

Goal: test whether adding real anomalies teaches generic anomaly localization or just overfits to seen defect morphology.

Hazelnut defect types: `crack`, `cut`, `hole`, `print`.

Run two families of experiments:

1. `B1_one_defect_type_in`

Fine-tune/train using real anomalies from only one defect type, for example only `crack`. Evaluate on all defect types.

Run separately for:

- only `crack`;
- only `cut`;
- only `hole`;
- only `print`.

2. `B2_leave_one_defect_type_out`

Fine-tune/train using all real anomaly types except one. Evaluate especially on the held-out type.

Run separately for:

- train without `crack`;
- train without `cut`;
- train without `hole`;
- train without `print`.

For fairness, keep the number of real anomaly images comparable across runs. If one variant has more available images, either subsample or report both fixed-count and all-available versions.

Interpretation:

- If held-out defect types improve too, real anomalies are improving generic anomaly representation.
- If seen types improve but held-out types degrade, the method is overfitting to defect morphology.
- If image metrics improve but pixel metrics drop, the classification branch is adapting but localization is being damaged.

## Experiment C: synthetic segmentation metric as a proxy

Goal: estimate whether segmentation quality on synthetic anomalies is a useful proxy for segmentation quality on real anomalies.

Motivation: in weak-supervised settings real anomaly masks may be unavailable, so real segmentation quality can only be inspected visually. A synthetic segmentation benchmark built from held-out normal images and generated masks could still be useful as a training diagnostic, early-stopping fallback, or regression check.

Protocol:

- split normal images into `normal_train`, `normal_val`, and `normal_synthetic_test`;
- train only on `normal_train`;
- evaluate synthetic segmentation metrics on `normal_val` and `normal_synthetic_test`;
- when real masks are available for a reference dataset, also evaluate real `pixel_aupro`, pixel ROC AUC, and pixel AP on the real validation/test split;
- log both metric families after each epoch or checkpoint.

Analysis:

- compare synthetic segmentation metrics against real mask metrics across epochs;
- compute rank/linear correlation between synthetic metrics and real metrics;
- check whether the epoch selected by synthetic metrics is close to the epoch selected by real mask metrics;
- inspect failure cases where synthetic quality improves while real anomaly localization gets worse.

Important naming rule: synthetic metrics must be named explicitly, for example `synthetic_pixel_aupro`, `synthetic_pixel_roc_auc`, `synthetic_pixel_ap`, to avoid confusing them with real MVTec-style mask metrics.

Success criterion: synthetic metrics do not need to match real metrics exactly, but they should preserve useful ordering across checkpoints. If they correlate poorly, keep them only as a sanity/regression diagnostic, not as an early-stopping target.

## Suggested implementation order

1. Add per-defect metric aggregation to the current evaluation if it is not present yet.
2. Implement `A3_two_stage_mixed_finetune_with_synthetic` first. It is the closest to the current pipeline and directly addresses the observed localization drop.
3. Sweep conservative segmentation fine-tune settings for A3:
   - LR: `2e-6`, `5e-6`, `1e-5`;
   - epochs: `2`, `4`, `6`;
   - weak map loss weight: `0`, `0.1`, `0.25`;
   - synthetic SSNLoss weight: `0.5`, `1.0`.
4. Only after A3 works, compare A1 and A2. They require more changes to the base training loop.
5. Run B1/B2 using the best A variant.
6. Run Experiment C on a dataset where real masks exist, then decide whether synthetic segmentation metrics are reliable enough for weak-supervised runs without masks.

## Loss design notes

Avoid updating the segmentation map only through image-level classification loss. The current metrics suggest this damages localization.

For weak real anomalies, prefer conservative losses:

- keep image-level BCE/Focal loss on `pred_score`;
- use direct weak map loss only with a small weight or after warmup;
- consider not applying positive top-k loss to real anomalies in the first ablation;
- for normal images, penalize only confident false-positive map pixels instead of all pixels, for example logits/probabilities above a threshold;
- keep synthetic anomalies with masks in the same fine-tune stage to preserve localization supervision.

The key comparison is not whether the training loss goes down. The key comparison is whether pixel AUPRO/AP stays stable or improves on holdout defects.
