# FastFlow MVTec comparison, try 1

Date: 2026-07-10

## Protocol

- Categories: `metal_nut` and `hazelnut`.
- Backbones: `resnet18` and `deit_base_distilled_patch16_384`.
- Input size: 384 x 384 for both backbones.
- Training data: only MVTec `train/good`; 10% is held out for normal validation loss.
- Labeled validation/test: the official MVTec test set is split 50/50 with defect-type stratification.
- `metal_nut/flip` is excluded; `good`, `bent`, `scratch`, and `color` are retained.
- `top_k_pixels` is selected on labeled validation by image ROC AUC, then AP, Cohen's d, and smaller k.
- The threshold is the 95th percentile of labeled-validation normal scores.
- Seed: 42. Training is deterministic where supported by PyTorch.

## Held-out test results

| Category | Backbone | Top-k pixels | Image ROC AUC | Image AP | Balanced accuracy | FPR | FNR | Pixel ROC AUC* | Pixel AP* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal_nut | ResNet18 | 1,024 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9618 | 0.5615 |
| metal_nut | DeiT | 1,024 | 0.9948 | 0.9984 | 0.8636 | 0.2727 | 0.0000 | 0.9727 | 0.6851 |
| hazelnut | ResNet18 | 8,192 | 0.9771 | 0.9872 | 0.8857 | 0.0000 | 0.2286 | 0.9420 | 0.4720 |
| hazelnut | DeiT | 4,096 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.9805 | 0.6019 |

\* The current notebook computes pixel metrics only on anomalous images. These values are useful for the internal paired comparison, but they are not directly comparable with the standard MVTec protocol that includes good images as all-zero masks.

## Findings

1. DeiT is not systematically worse than ResNet18. On `metal_nut`, image ranking is effectively tied: ResNet18 ROC AUC is 1.0000 and DeiT is 0.9948. On `hazelnut`, DeiT improves ROC AUC from 0.9771 to 1.0000 and removes eight ResNet18 false negatives: two `crack`, three `cut`, and three `hole` samples.
2. DeiT produces better pixel ranking on both categories. Pixel ROC AUC/AP improve from 0.9618/0.5615 to 0.9727/0.6851 on `metal_nut`, and from 0.9420/0.4720 to 0.9805/0.6019 on `hazelnut`. Interpret this only as an internal comparison because of the pixel-metric caveat above.
3. The validation-selected top-k transfers well to held-out test. ResNet18 selects 1,024 pixels on `metal_nut` and 8,192 on `hazelnut`; DeiT selects 1,024 and 4,096 respectively. All selected values remain within 0.01 ROC AUC of the best diagnostic test value. This supports category- and backbone-specific top-k calibration rather than one global fixed value.
4. Ranking and threshold calibration must be evaluated separately. `metal_nut` DeiT has ROC AUC 0.9948 but three false positives and balanced accuracy 0.8636. The threshold is estimated from only 11 validation-good images, so a 95th-percentile estimate is unstable despite excellent ranking.
5. Paired stratified bootstrap over the fixed held-out split gives a DeiT-minus-ResNet18 ROC AUC difference of -0.0052 on `metal_nut` (bootstrap 95% interval approximately [-0.0234, 0.0000]) and +0.0229 on `hazelnut` ([0.0000, 0.0614]). These intervals describe sampling uncertainty on this split, not training-seed uncertainty.
6. Both DeiT runs and the `hazelnut` ResNet18 run reach their best normal validation loss at the final epoch. This suggests that flow likelihood had not fully plateaued, but the image metrics are already saturated for DeiT; extending MVTec training alone is therefore low priority.

## Limitations

- Only one training/split seed was evaluated.
- Only two MVTec categories were evaluated.
- Half of the official MVTec test set is used as labeled validation, so the numbers are an internal held-out comparison, not canonical full-test benchmark results.
- The backbones use architecture-specific optimization settings. This is a comparison of practical configured systems, not an isolated backbone-only ablation.
- Pixel metrics exclude good images in this run.
- Threshold calibration uses few normal validation samples, especially for `metal_nut`.

## Recommended next experiments

1. Fix pixel evaluation to include good images with zero masks and add AUPRO if a standard MVTec comparison is required.
2. Separate training seed from split seed. Run at least two additional training seeds and report mean, standard deviation, and per-seed top-k. This is the minimum check before claiming stable DeiT non-inferiority.
3. Evaluate threshold calibration over repeated splits or a larger held-out normal calibration set. Do not tune threshold on held-out test.
4. Move to the printer dataset after the minimal multi-seed check. Use a leak-free train/calibration/test split, select top-k and threshold only on calibration, and keep test locked until the configuration is frozen.
