import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from fastflow_printer_pipeline import (
    PRINTER_CONFIGS,
    load_approved_manifest,
    load_experiment_result,
    resolve_project_root,
    run_experiment,
)


DEFAULT_CONFIGS = ["resnet18_256", "resnet18_384", "deit_base_distilled_384"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run staged printer FastFlow experiments.")
    parser.add_argument(
        "--stage",
        choices=("train-calibrate", "test", "load"),
        default="train-calibrate",
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025])
    parser.add_argument("--try-number", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/processed_printer_dataset_384"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(
            "experiments/printer/dataset_v384_audit/printer_split_v2_final.csv"
        ),
    )
    parser.add_argument(
        "--split-config-path",
        type=Path,
        default=Path("configs/printer_split_v2_final.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/printer/printer384_v2_final_summaries"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = set(args.configs) - set(PRINTER_CONFIGS)
    if unknown:
        raise KeyError(f"Unknown configs: {sorted(unknown)}")

    project_root = resolve_project_root()
    dataset_root = project_root / args.dataset_root
    manifest_path = project_root / args.manifest_path
    split_config_path = project_root / args.split_config_path
    experiments_root = project_root / "experiments" / "printer"

    _, split_config = load_approved_manifest(
        dataset_root,
        manifest_path,
        split_config_path,
        verify_hashes=not args.skip_hash_verification,
    )

    results = []
    for seed in args.seeds:
        for config_name in args.configs:
            config = PRINTER_CONFIGS[config_name]
            if args.stage == "load":
                result = load_experiment_result(
                    experiments_root,
                    config,
                    args.try_number,
                    seed,
                )
            else:
                result = run_experiment(
                    config=config,
                    seed=seed,
                    dataset_root=dataset_root,
                    manifest_path=manifest_path,
                    split_config_path=split_config_path,
                    experiments_root=experiments_root,
                    try_number=args.try_number,
                    run_training=args.stage == "train-calibrate",
                    run_calibration=args.stage == "train-calibrate",
                    run_test=args.stage == "test",
                    allow_overwrite=args.allow_overwrite,
                    deterministic=True,
                    verify_manifest_hashes=False,
                    bootstrap_iterations=args.bootstrap_iterations,
                )
            results.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = pd.DataFrame(results)
    stage_name = args.stage.replace("-", "_")
    config_slug = "-".join(args.configs)
    output_dir = project_root / args.output_dir
    output_path = output_dir / (
        f"fastflow_{split_config['version']}_{config_slug}_{stage_name}_"
        f"try_{args.try_number}.csv"
    )
    if output_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite summary: {output_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "configs": args.configs,
                "seeds": args.seeds,
                "runs": len(results),
                "summary": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
