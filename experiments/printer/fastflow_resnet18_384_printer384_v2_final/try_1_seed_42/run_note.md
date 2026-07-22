# resnet18_384 / seed 42

Status at creation: planned train + calibration; test locked.
Role: resolution-matched ResNet18 control for DeiT-384.
Split: printer_384_v2_final.
Manifest SHA-256: `aac844a06b740658ffcb756f033efd1722a0258f50b16c3e9dce0ae38d431317`.
Pipeline SHA-256: `676770e70f971b0a933cab832004a6ebe95d027918523e09bfaed6d3d28627d3`.
Git commit: `1a7038aa064a4d08d80603bba1d1c0e4d5d63916`.
Git worktree dirty: `False`.

Training parameters:

- input: 384x384
- epochs: 30
- flow_steps: 10
- learning_rate: 0.001
- weight_decay: 0.001
- batch_size: 8
- grad_clip_norm: None
- augmentation: none

Top-k and threshold are selected only on calibration. Test must not be
read until the candidate configurations are frozen.

## Outcome

Status: failed before model construction and before the first training batch.

`timm`, called internally by Anomalib `FastflowModel(pre_trained=True)`, tried
to query Hugging Face through an unavailable local proxy. No epoch, training
batch, calibration inference, or test inference ran. The original artifacts
are preserved. Exact failure facts are in `failure.json`.

Both required backbones were then successfully constructed from the existing
local Hugging Face cache with process-local `HF_HUB_OFFLINE=1`. The retry must
use a new `try_2` directory and must not alter proxy/VPN settings.
