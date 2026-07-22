import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastflow_printer_pipeline import file_sha256


SOURCE_FILES = (
    "run_config.json",
    "execution_provenance.json",
    "train_history.csv",
    "training_summary.json",
    "calibration_selection.json",
    "calibration_metrics.json",
    "calibration_top_k_sweep.csv",
    "calibration_scores.csv",
)
OUTPUT_FILES = (
    "calibration_report.json",
    "calibration_report.md",
    "calibration_training_curves.png",
    "calibration_top_k_sweep.png",
    "calibration_score_distribution.png",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_sources(log_dir):
    missing = [name for name in SOURCE_FILES if not (log_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts in {log_dir}: {missing}")

    run_config = load_json(log_dir / "run_config.json")
    history = pd.read_csv(log_dir / "train_history.csv")
    training = load_json(log_dir / "training_summary.json")
    selection = load_json(log_dir / "calibration_selection.json")
    metrics = load_json(log_dir / "calibration_metrics.json")
    sweep = pd.read_csv(log_dir / "calibration_top_k_sweep.csv")
    scores = pd.read_csv(log_dir / "calibration_scores.csv")

    expected_epochs = int(run_config["config"]["num_epochs"])
    if len(history) != expected_epochs:
        raise ValueError(f"Expected {expected_epochs} history rows, got {len(history)}")
    best_index = int(history["normal_val_loss"].idxmin())
    best_row = history.loc[best_index]
    if int(best_row["epoch"]) != int(training["best_epoch"]):
        raise ValueError("training_summary best_epoch disagrees with train_history")
    if not np.isclose(
        best_row["normal_val_loss"],
        training["best_normal_val_loss"],
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError("training_summary best loss disagrees with train_history")

    selected = sweep.loc[sweep["selected"].astype(bool)]
    if len(selected) != 1:
        raise ValueError("Top-k sweep must contain exactly one selected row")
    selected_row = selected.iloc[0]
    selected_k = int(selection["selected_top_k_pixels"])
    if int(selected_row["top_k_pixels"]) != selected_k:
        raise ValueError("calibration_selection disagrees with top-k sweep")
    if selected_k != int(metrics["selected_top_k_pixels"]):
        raise ValueError("calibration_selection disagrees with calibration_metrics")
    primary = "source_group_balanced_tile_roc_auc"
    if not np.isclose(
        selected_row[primary],
        metrics[f"calibration_{primary}"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Selected top-k metric disagrees with calibration_metrics")

    expected_rows = int(metrics["calibration_num_good"]) + int(
        metrics["calibration_num_anomaly"]
    )
    if len(scores) != expected_rows:
        raise ValueError("calibration score count disagrees with calibration_metrics")
    return run_config, history, training, selection, metrics, sweep, scores


def plot_training(history, output_path):
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["normal_val_loss"], label="normal val")
    axes[0].set_ylabel("FastFlow loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(history["epoch"], history["learning_rate"], color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_top_k(sweep, selection, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        sweep["top_k_fraction"],
        sweep["source_group_balanced_tile_roc_auc"],
        marker="o",
        label="source-balanced tile ROC AUC",
    )
    axis.plot(
        sweep["top_k_fraction"],
        sweep["object_roc_auc"],
        marker="o",
        label="object ROC AUC",
    )
    axis.plot(
        sweep["top_k_fraction"],
        sweep["tile_roc_auc"],
        marker="o",
        label="tile ROC AUC",
    )
    axis.axvline(
        selection["selected_top_k_fraction"],
        color="black",
        linestyle="--",
        label="selected top-k",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Top-k fraction")
    axis.set_ylabel("ROC AUC")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_scores(scores, threshold, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, color, name in ((0, "tab:blue", "good"), (1, "tab:red", "anomaly")):
        values = scores.loc[scores["label"] == label, "score"].sort_values().to_numpy()
        axis.scatter(values, np.full(len(values), label), alpha=0.8, color=color, label=name)
    axis.axvline(threshold, color="black", linestyle="--", label="normal q95 threshold")
    axis.set_yticks([0, 1], labels=["good", "anomaly"])
    axis.set_xlabel("Selected top-k anomaly score")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_report(log_dir, allow_overwrite=False):
    log_dir = Path(log_dir)
    existing = [name for name in OUTPUT_FILES if (log_dir / name).exists()]
    if existing and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite report artifacts: {existing}")
    run_config, history, training, selection, metrics, sweep, scores = validate_sources(
        log_dir
    )

    plot_training(history, log_dir / "calibration_training_curves.png")
    plot_top_k(sweep, selection, log_dir / "calibration_top_k_sweep.png")
    plot_scores(scores, selection["threshold"], log_dir / "calibration_score_distribution.png")

    source_hashes = {name: file_sha256(log_dir / name) for name in SOURCE_FILES}
    report = {
        "validation_status": "passed",
        "config_name": run_config["config"]["name"],
        "seed": run_config["seed"],
        "best_epoch": training["best_epoch"],
        "best_normal_val_loss": training["best_normal_val_loss"],
        "selected_top_k_pixels": selection["selected_top_k_pixels"],
        "selected_top_k_fraction": selection["selected_top_k_fraction"],
        "calibration_primary_roc_auc": metrics[
            "calibration_source_group_balanced_tile_roc_auc"
        ],
        "calibration_object_roc_auc": metrics["calibration_object_roc_auc"],
        "calibration_source_image_roc_auc": metrics[
            "calibration_source_image_roc_auc"
        ],
        "calibration_balanced_accuracy": metrics[
            "calibration_source_group_balanced_tile_balanced_accuracy"
        ],
        "calibration_fpr": metrics["calibration_source_group_balanced_tile_fpr"],
        "calibration_fnr": metrics["calibration_source_group_balanced_tile_fnr"],
        "source_artifact_sha256": source_hashes,
    }
    (log_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {report['config_name']} / seed {report['seed']} calibration result",
        "",
        "Source validation: passed.",
        "",
        f"- best epoch: {report['best_epoch']}",
        f"- best normal-val loss: {report['best_normal_val_loss']}",
        f"- selected top-k: {report['selected_top_k_pixels']} "
        f"({report['selected_top_k_fraction']:.6f})",
        f"- source-balanced tile ROC AUC: {report['calibration_primary_roc_auc']:.6f}",
        f"- object ROC AUC: {report['calibration_object_roc_auc']:.6f}",
        f"- source-image ROC AUC: {report['calibration_source_image_roc_auc']:.6f}",
        f"- balanced accuracy: {report['calibration_balanced_accuracy']:.6f}",
        f"- FPR: {report['calibration_fpr']:.6f}",
        f"- FNR: {report['calibration_fnr']:.6f}",
        "",
        "Figures:",
        "",
        "- `calibration_training_curves.png`",
        "- `calibration_top_k_sweep.png`",
        "- `calibration_score_distribution.png`",
        "",
        "Machine-readable values and source hashes: `calibration_report.json`.",
    ]
    (log_dir / "calibration_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and report one calibration run.")
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args.log_dir, allow_overwrite=args.allow_overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
