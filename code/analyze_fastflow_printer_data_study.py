import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fastflow_printer_pipeline import group_balanced_weights, multilevel_ranking_metrics


PRIMARY = "source_group_balanced_tile_roc_auc"
METADATA_COLUMNS = ["path", "source_group", "object_group", "source_date", "label"]
SEEDS = (42, 123, 2025)
POST_SELECTION_SEEDS = (123, 2025)

PAIRED_RUNS = {
    "resnet18_384": {
        42: "fastflow_resnet18_384_printer384_v2_final/try_2_seed_42",
        123: "fastflow_resnet18_384_printer384_v2_final/try_3_seed_123",
        2025: "fastflow_resnet18_384_printer384_v2_final/try_4_seed_2025",
    },
    "deit_no_aug": {
        42: "fastflow_deit_base_distilled_384_printer384_v2_final/try_1_seed_42",
        123: "fastflow_deit_base_distilled_384_printer384_v2_final/try_2_seed_123",
        2025: "fastflow_deit_base_distilled_384_printer384_v2_final/try_3_seed_2025",
    },
    "deit_strong_aug": {
        42: "fastflow_deit_data_strong_aug_1000_printer384_v2_data_study/try_1_seed_42",
        123: "fastflow_deit_data_strong_aug_1000_printer384_v2_data_study/try_2_seed_123",
        2025: "fastflow_deit_data_strong_aug_1000_printer384_v2_data_study/try_3_seed_2025",
    },
}

FIRST_PASS_RUNS = {
    "deit_no_aug": PAIRED_RUNS["deit_no_aug"][42],
    "deit_mild_photo_1000": (
        "fastflow_deit_data_mild_photo_1000_printer384_v2_data_study/"
        "try_2_seed_42"
    ),
    "deit_strong_aug_1000": PAIRED_RUNS["deit_strong_aug"][42],
    "deit_date_balanced_500": (
        "fastflow_deit_data_balanced_500_printer384_v2_data_study/"
        "try_1_seed_42"
    ),
    "deit_all_1671": (
        "fastflow_deit_data_all_1671_printer384_v2_data_study/"
        "try_1_seed_42"
    ),
    "deit_object_uniform_1000": (
        "fastflow_deit_data_object_uniform_1000_printer384_v2_data_study/"
        "try_1_seed_42"
    ),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(experiments_root, name, seed, relative_path):
    run_dir = experiments_root / relative_path
    required = (
        "calibration_metrics.json",
        "calibration_scores.csv",
        "calibration_top_k_scores.csv",
        "training_summary.json",
        "train_history.csv",
        "run_config.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing {missing}")

    metrics = read_json(run_dir / "calibration_metrics.json")
    training = read_json(run_dir / "training_summary.json")
    run_config = read_json(run_dir / "run_config.json")
    scores = pd.read_csv(run_dir / "calibration_scores.csv")
    top_k_scores = pd.read_csv(run_dir / "calibration_top_k_scores.csv")
    history = pd.read_csv(run_dir / "train_history.csv")

    if scores["path"].duplicated().any():
        raise ValueError(f"Duplicate calibration paths in {run_dir}")
    if not scores[METADATA_COLUMNS].equals(top_k_scores[METADATA_COLUMNS]):
        raise ValueError(f"Selected and top-k score rows differ in {run_dir}")

    selected = multilevel_ranking_metrics(scores, "score")
    saved_keys = {
        PRIMARY: f"calibration_{PRIMARY}",
        "object_roc_auc": "calibration_object_roc_auc",
        "source_image_roc_auc": "calibration_source_image_roc_auc",
    }
    for key, saved_key in saved_keys.items():
        if not np.isclose(selected[key], metrics[saved_key]):
            raise ValueError(f"Saved {saved_key} does not match raw scores in {run_dir}")

    num_pixels = int(metrics["num_pixels"])
    full_column = f"score_top_{num_pixels}"
    full_scores = top_k_scores[METADATA_COLUMNS].copy()
    full_scores["score"] = top_k_scores[full_column]
    full_metrics = multilevel_ranking_metrics(full_scores, "score")
    scores = scores.sort_values("path").reset_index(drop=True)
    full_scores = full_scores.sort_values("path").reset_index(drop=True)

    train_subset = run_config.get("train_subset", {})
    row = {
        "config_name": name,
        "seed": int(seed),
        "selected_top_k_pixels": int(metrics["selected_top_k_pixels"]),
        "selected_top_k_fraction": float(metrics["selected_top_k_fraction"]),
        "primary_roc_auc": float(selected[PRIMARY]),
        "full_map_primary_roc_auc": float(full_metrics[PRIMARY]),
        "object_roc_auc": float(selected["object_roc_auc"]),
        "source_image_roc_auc": float(selected["source_image_roc_auc"]),
        "balanced_accuracy": float(
            metrics["calibration_source_group_balanced_tile_balanced_accuracy"]
        ),
        "fpr": float(metrics["calibration_source_group_balanced_tile_fpr"]),
        "fnr": float(metrics["calibration_source_group_balanced_tile_fnr"]),
        "best_epoch": int(training["best_epoch"]),
        "best_normal_val_loss": float(training["best_normal_val_loss"]),
        "training_duration_seconds": float(
            training.get("training_duration_seconds", np.nan)
        ),
        "peak_cuda_memory_mib": float(
            training.get("peak_cuda_memory_mib", np.nan)
        ),
        "gradient_clip_fraction_max": (
            float(history["gradient_clip_fraction"].max())
            if "gradient_clip_fraction" in history
            else np.nan
        ),
        "train_tiles": int(
            train_subset.get("selected_tiles", run_config["split_counts"]["train"])
        ),
        "train_source_groups": int(train_subset.get("source_groups", 212)),
        "train_object_groups": int(train_subset.get("object_groups", 687)),
        "augmentation": run_config["config"]["augmentation"],
        "run_dir": run_dir.as_posix(),
    }
    return {"row": row, "scores": scores, "full_scores": full_scores}


def validate_calibration_rows(runs):
    reference_name, reference = next(iter(runs.items()))
    reference_rows = reference["scores"][METADATA_COLUMNS]
    for name, run in runs.items():
        if not reference_rows.equals(run["scores"][METADATA_COLUMNS]):
            raise ValueError(f"Calibration rows differ: {reference_name} vs {name}")


def validate_train_paths(project_root):
    old_manifest = pd.read_csv(
        project_root
        / "experiments/printer/dataset_v384_audit/printer_split_v2_final.csv"
    )
    study_manifest = pd.read_csv(
        project_root
        / "experiments/printer/dataset_v384_audit/printer_split_v2_data_study.csv"
    )
    old_paths = sorted(old_manifest.loc[old_manifest["split"] == "train", "path"])
    ranks = pd.to_numeric(study_manifest["train_rank_date_balanced"], errors="coerce")
    study_paths = sorted(
        study_manifest.loc[
            (study_manifest["split"] == "train") & (ranks <= 1000),
            "path",
        ]
    )
    if old_paths != study_paths:
        raise ValueError("Old baseline and data-study 1000-tile train paths differ")
    digest = hashlib.sha256("\n".join(old_paths).encode("utf-8")).hexdigest()
    return len(old_paths), digest


def primary_metric(rows):
    return float(
        roc_auc_score(
            rows["label"],
            rows["score"],
            sample_weight=group_balanced_weights(rows["source_group"]),
        )
    )


def group_score_arrays(rows):
    return {
        source_group: group["score"].to_numpy(dtype=float)
        for source_group, group in rows.groupby("source_group", sort=False)
    }


def primary_from_group_draw(score_arrays, group_draw):
    scores = []
    labels = []
    weights = []
    for label, source_group in group_draw:
        group_scores = score_arrays[source_group]
        scores.append(group_scores)
        labels.append(np.full(len(group_scores), label, dtype=int))
        weights.append(np.full(len(group_scores), 1.0 / len(group_scores)))
    return float(
        roc_auc_score(
            np.concatenate(labels),
            np.concatenate(scores),
            sample_weight=np.concatenate(weights),
        )
    )


def paired_bootstrap(reference, candidate, seeds, score_key, iterations, rng):
    template = reference[seeds[0]][score_key]
    groups_by_label = {
        label: template.loc[
            template["label"] == label, "source_group"
        ].drop_duplicates().to_numpy()
        for label in (0, 1)
    }
    reference_arrays = {
        seed: group_score_arrays(reference[seed][score_key]) for seed in seeds
    }
    candidate_arrays = {
        seed: group_score_arrays(candidate[seed][score_key]) for seed in seeds
    }
    full_draw = [
        (label, source_group)
        for label, groups in groups_by_label.items()
        for source_group in groups
    ]
    for seed in seeds:
        expected = primary_metric(reference[seed][score_key])
        observed = primary_from_group_draw(reference_arrays[seed], full_draw)
        if not np.isclose(expected, observed):
            raise ValueError("Array and dataframe primary metrics differ")
    differences = []
    for _ in range(iterations):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        group_draw = [
            (label, source_group)
            for label, groups in groups_by_label.items()
            for source_group in rng.choice(groups, size=len(groups), replace=True)
        ]
        seed_differences = []
        for seed in sampled_seeds:
            seed = int(seed)
            seed_differences.append(
                primary_from_group_draw(candidate_arrays[seed], group_draw)
                - primary_from_group_draw(reference_arrays[seed], group_draw)
            )
        differences.append(np.mean(seed_differences))
    return np.quantile(differences, [0.025, 0.975])


def comparison_rows(
    reference_name,
    candidate_name,
    paired_runs,
    seeds,
    iterations,
    rng,
):
    rows = []
    for score_key, score_variant in (
        ("scores", "selected_top_k"),
        ("full_scores", "fixed_full_map"),
    ):
        deltas = []
        for seed in seeds:
            reference_value = primary_metric(paired_runs[reference_name][seed][score_key])
            candidate_value = primary_metric(paired_runs[candidate_name][seed][score_key])
            deltas.append(candidate_value - reference_value)
        ci_low, ci_high = paired_bootstrap(
            paired_runs[reference_name],
            paired_runs[candidate_name],
            tuple(seeds),
            score_key,
            iterations,
            rng,
        )
        rows.append(
            {
                "reference": reference_name,
                "candidate": candidate_name,
                "seed_scope": (
                    "post_selection_123_2025"
                    if tuple(seeds) == POST_SELECTION_SEEDS
                    else "all_42_123_2025"
                ),
                "score_variant": score_variant,
                "seeds": len(seeds),
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "positive_seed_deltas": int(np.sum(np.asarray(deltas) > 0)),
            }
        )
    return rows


def create_dashboard(first_pass, paired, output_path):
    colors = {
        "resnet18_384": "#3b6ea8",
        "deit_no_aug": "#666666",
        "deit_strong_aug": "#c84b31",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ordered = first_pass.sort_values("primary_roc_auc")
    axes[0, 0].barh(ordered["config_name"], ordered["primary_roc_auc"], color="#4f7f68")
    axes[0, 0].axvline(
        first_pass.loc[
            first_pass["config_name"] == "deit_no_aug", "primary_roc_auc"
        ].iloc[0],
        color="#555555",
        linestyle="--",
        label="no-aug baseline",
    )
    axes[0, 0].set(xlabel="Calibration primary ROC AUC", title="First pass, seed 42")
    axes[0, 0].set_xlim(0.84, 0.97)
    axes[0, 0].legend()

    for config_name, frame in paired.groupby("config_name"):
        axes[0, 1].plot(
            frame["seed"].astype(str),
            frame["primary_roc_auc"],
            marker="o",
            label=config_name,
            color=colors[config_name],
        )
    axes[0, 1].set(xlabel="Seed", ylabel="Primary ROC AUC", title="Paired calibration")
    axes[0, 1].set_ylim(0.84, 0.98)
    axes[0, 1].legend()

    strong = paired.loc[paired["config_name"] == "deit_strong_aug"].set_index("seed")
    for index, reference_name in enumerate(("deit_no_aug", "resnet18_384")):
        reference = paired.loc[paired["config_name"] == reference_name].set_index("seed")
        delta = strong["primary_roc_auc"] - reference["primary_roc_auc"]
        axes[1, 0].bar(
            np.arange(len(SEEDS)) + (index - 0.5) * 0.32,
            delta.loc[list(SEEDS)],
            width=0.32,
            label=f"strong - {reference_name}",
        )
    axes[1, 0].axhline(0, color="#222222", linewidth=1)
    axes[1, 0].set_xticks(np.arange(len(SEEDS)), [str(seed) for seed in SEEDS])
    axes[1, 0].set(xlabel="Seed", ylabel="Primary ROC AUC delta", title="Paired deltas")
    axes[1, 0].legend()

    metric_columns = ["primary_roc_auc", "object_roc_auc", "source_image_roc_auc"]
    metric_labels = ["Primary", "Object", "Source"]
    x = np.arange(len(metric_columns))
    for index, config_name in enumerate(colors):
        means = paired.loc[paired["config_name"] == config_name, metric_columns].mean()
        axes[1, 1].bar(
            x + (index - 1) * 0.25,
            means,
            width=0.25,
            label=config_name,
            color=colors[config_name],
        )
    axes[1, 1].set_xticks(x, metric_labels)
    axes[1, 1].set(ylabel="Mean ROC AUC over 3 seeds", title="Aggregation levels")
    axes[1, 1].set_ylim(0.82, 1.0)
    axes[1, 1].legend()

    fig.suptitle("FastFlow printer DeiT data study (calibration only)", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze printer DeiT data study.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/printer/printer384_v2_data_study_analysis"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    experiments_root = project_root / "experiments/printer"
    train_count, train_paths_sha256 = validate_train_paths(project_root)

    paired_runs = {
        config_name: {
            seed: load_run(experiments_root, config_name, seed, relative_path)
            for seed, relative_path in runs.items()
        }
        for config_name, runs in PAIRED_RUNS.items()
    }
    flattened = {
        f"{config_name}:{seed}": run
        for config_name, runs in paired_runs.items()
        for seed, run in runs.items()
    }
    validate_calibration_rows(flattened)

    first_pass_runs = {
        name: load_run(experiments_root, name, 42, relative_path)
        for name, relative_path in FIRST_PASS_RUNS.items()
    }
    validate_calibration_rows(first_pass_runs)
    first_pass = pd.DataFrame([run["row"] for run in first_pass_runs.values()])
    baseline_auc = first_pass.loc[
        first_pass["config_name"] == "deit_no_aug", "primary_roc_auc"
    ].iloc[0]
    first_pass["delta_to_no_aug"] = first_pass["primary_roc_auc"] - baseline_auc

    paired = pd.DataFrame(
        [
            run["row"]
            for config_runs in paired_runs.values()
            for run in config_runs.values()
        ]
    )
    by_config = (
        paired.groupby("config_name", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            primary_mean=("primary_roc_auc", "mean"),
            primary_sample_std=("primary_roc_auc", "std"),
            object_mean=("object_roc_auc", "mean"),
            source_mean=("source_image_roc_auc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            top_k_unique=("selected_top_k_pixels", "nunique"),
            training_duration_mean_seconds=("training_duration_seconds", "mean"),
        )
        .sort_values("primary_mean", ascending=False)
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = []
    for reference in ("deit_no_aug", "resnet18_384"):
        for seed_scope in (SEEDS, POST_SELECTION_SEEDS):
            comparisons.extend(
                comparison_rows(
                    reference,
                    "deit_strong_aug",
                    paired_runs,
                    seed_scope,
                    args.bootstrap_iterations,
                    rng,
                )
            )
    comparisons = pd.DataFrame(comparisons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "first_pass": args.output_dir / "first_pass_seed42.csv",
        "paired_runs": args.output_dir / "paired_runs.csv",
        "by_config": args.output_dir / "paired_summary_by_config.csv",
        "comparisons": args.output_dir / "paired_comparisons.csv",
        "dashboard": args.output_dir / "data_study_dashboard.png",
        "summary": args.output_dir / "data_study_summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    first_pass.to_csv(outputs["first_pass"], index=False)
    paired.to_csv(outputs["paired_runs"], index=False)
    by_config.to_csv(outputs["by_config"], index=False)
    comparisons.to_csv(outputs["comparisons"], index=False)
    create_dashboard(first_pass, paired, outputs["dashboard"])
    summary = {
        "calibration_rows_identical": True,
        "calibration_tiles": int(len(next(iter(flattened.values()))["scores"])),
        "calibration_source_groups": int(
            next(iter(flattened.values()))["scores"]["source_group"].nunique()
        ),
        "old_and_study_train_paths_identical": True,
        "train_tiles": train_count,
        "train_paths_sha256": train_paths_sha256,
        "test_read_by_analysis": False,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "notes": [
            "Seed 42 selected the winner from five variants and is exploratory.",
            "Seeds 123 and 2025 are the post-selection replication set.",
            "Bootstrap resamples source groups within class and seeds.",
            "Bootstrap does not correct top-k tuning or first-pass selection bias.",
        ],
    }
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(first_pass[["config_name", "primary_roc_auc", "delta_to_no_aug"]].to_string(index=False))
    print(by_config.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
