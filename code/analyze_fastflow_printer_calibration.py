import argparse
import json
from pathlib import Path

import pandas as pd

from fastflow_printer_pipeline import PRINTER_CONFIGS, experiment_log_dir


PRIMARY_METRIC = "calibration_source_group_balanced_tile_roc_auc"
DEFAULT_CONFIGS = ["resnet18_256", "resnet18_384", "deit_base_distilled_384"]


def load_calibration_run(experiments_root, config_name, try_number, seed):
    config = PRINTER_CONFIGS[config_name]
    log_dir = experiment_log_dir(experiments_root, config, try_number, seed)
    paths = {
        "metrics": log_dir / "calibration_metrics.json",
        "training": log_dir / "training_summary.json",
        "history": log_dir / "train_history.csv",
        "scores": log_dir / "calibration_scores.csv",
        "sweep": log_dir / "calibration_top_k_sweep.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing calibration artifacts: {missing}")

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    training = json.loads(paths["training"].read_text(encoding="utf-8"))
    history = pd.read_csv(paths["history"])
    scores = pd.read_csv(paths["scores"])
    sweep = pd.read_csv(paths["sweep"])

    row = {
        "config_name": config_name,
        "seed": int(seed),
        "selected_top_k_pixels": metrics["selected_top_k_pixels"],
        "selected_top_k_fraction": metrics["selected_top_k_fraction"],
        PRIMARY_METRIC: metrics[PRIMARY_METRIC],
        "calibration_tile_roc_auc": metrics["calibration_tile_roc_auc"],
        "calibration_object_roc_auc": metrics["calibration_object_roc_auc"],
        "calibration_source_image_roc_auc": metrics[
            "calibration_source_image_roc_auc"
        ],
        "calibration_balanced_accuracy": metrics[
            "calibration_source_group_balanced_tile_balanced_accuracy"
        ],
        "calibration_fpr": metrics["calibration_source_group_balanced_tile_fpr"],
        "calibration_fnr": metrics["calibration_source_group_balanced_tile_fnr"],
        "best_epoch": training["best_epoch"],
        "best_normal_val_loss": training["best_normal_val_loss"],
        "final_train_loss": training["final_train_loss"],
        "final_normal_val_loss": training["final_normal_val_loss"],
        "gradient_clip_fraction_max": (
            history["gradient_clip_fraction"].max()
            if "gradient_clip_fraction" in history
            else float("nan")
        ),
        "gradient_norm_max_before_clip": (
            history["gradient_norm_max_before_clip"].max()
            if "gradient_norm_max_before_clip" in history
            else float("nan")
        ),
        "log_dir": str(log_dir),
    }

    source_scores = (
        scores.groupby(["source_group", "label"], as_index=False)
        .agg(source_max_score=("score", "max"), source_mean_score=("score", "mean"), tiles=("score", "size"))
        .assign(config_name=config_name, seed=int(seed))
    )
    sweep = sweep.assign(config_name=config_name, seed=int(seed))
    return row, source_scores, sweep


def summarize_configs(per_seed):
    return (
        per_seed.groupby("config_name", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            primary_mean=(PRIMARY_METRIC, "mean"),
            primary_std=(PRIMARY_METRIC, "std"),
            primary_min=(PRIMARY_METRIC, "min"),
            primary_max=(PRIMARY_METRIC, "max"),
            object_roc_auc_mean=("calibration_object_roc_auc", "mean"),
            source_image_roc_auc_mean=("calibration_source_image_roc_auc", "mean"),
            top_k_unique=("selected_top_k_pixels", "nunique"),
            top_k_fraction_min=("selected_top_k_fraction", "min"),
            top_k_fraction_max=("selected_top_k_fraction", "max"),
            best_epoch_mean=("best_epoch", "mean"),
            clip_fraction_max=("gradient_clip_fraction_max", "max"),
        )
        .sort_values("primary_mean", ascending=False)
    )


def write_csv(path, frame, allow_overwrite):
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze printer calibration runs.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments/printer"),
    )
    parser.add_argument("--try-number", type=int, default=1)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/printer/printer384_v2_final_summaries"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = set(args.configs) - set(PRINTER_CONFIGS)
    if unknown:
        raise KeyError(f"Unknown configs: {sorted(unknown)}")

    rows = []
    source_frames = []
    sweep_frames = []
    for config_name in args.configs:
        for seed in args.seeds:
            row, source_scores, sweep = load_calibration_run(
                args.experiments_root,
                config_name,
                args.try_number,
                seed,
            )
            rows.append(row)
            source_frames.append(source_scores)
            sweep_frames.append(sweep)

    per_seed = pd.DataFrame(rows)
    by_config = summarize_configs(per_seed)
    source_scores = pd.concat(source_frames, ignore_index=True)
    sweeps = pd.concat(sweep_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_seed": args.output_dir
        / f"fastflow_printer384_v2_final_calibration_per_seed_try_{args.try_number}.csv",
        "by_config": args.output_dir
        / f"fastflow_printer384_v2_final_calibration_by_config_try_{args.try_number}.csv",
        "source_scores": args.output_dir
        / f"fastflow_printer384_v2_final_calibration_source_scores_try_{args.try_number}.csv",
        "top_k_sweeps": args.output_dir
        / f"fastflow_printer384_v2_final_calibration_top_k_sweeps_try_{args.try_number}.csv",
    }
    frames = {
        "per_seed": per_seed,
        "by_config": by_config,
        "source_scores": source_scores,
        "top_k_sweeps": sweeps,
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite calibration reports: {existing}")
    for name, path in outputs.items():
        write_csv(path, frames[name], args.allow_overwrite)

    print(by_config.to_string(index=False))
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
