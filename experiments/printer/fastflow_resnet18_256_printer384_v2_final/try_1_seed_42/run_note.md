# resnet18_256 / try 1 / seed 42

Experiment number: 1.

Purpose: establish the historical 256x256 ResNet18 FastFlow control on the
approved `printer_384_v2_final` split before comparing it with the
resolution-matched ResNet18-384 and DeiT-384 systems.

Stage executed: train + calibration. Locked test was not read.

Key settings:

- backbone: `resnet18`
- input: 256x256
- epochs: 14
- flow steps: 10
- learning rate: 1e-3
- weight decay: 1e-3
- batch size: 8
- augmentation: none
- gradient clipping: none
- seed: 42

Top-k and the score threshold were selected only on calibration. Exact
configuration, split counts and manifest hashes are in `run_config.json`.

The run started from an uncommitted working tree based on Git commit
`7c1ced12ff88`. Exact execution-file hashes are recorded in
`execution_provenance.json`. No later commit hash is retroactively attributed
to this run.
