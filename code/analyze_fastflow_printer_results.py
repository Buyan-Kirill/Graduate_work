import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fastflow_printer_pipeline import (
    PRINTER_CONFIGS,
    experiment_log_dir,
    multilevel_ranking_metrics,
)


PRIMARY_METRIC = "source_group_balanced_tile_roc_auc"
METADATA_COLUMNS = ["path", "source_group", "object_group", "source_date", "label"]


def load_run(experiments_root, config_name, try_number, seed):
    config = PRINTER_CONFIGS[config_name]
    log_dir = experiment_log_dir(experiments_root, config, try_number, seed)
    scores_path = log_dir / "test_scores.csv"
    metrics_path = log_dir / "test_metrics.json"
    run_config_path = log_dir / "run_config.json"
    missing = [
        path for path in (scores_path, metrics_path, run_config_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")

    scores = pd.read_csv(scores_path)
    required_columns = set(METADATA_COLUMNS) | {"score"}
    if required_columns - set(scores.columns):
        raise ValueError(f"Missing test score columns in {scores_path}")
    if scores["path"].duplicated().any():
        raise ValueError(f"Duplicate test paths in {scores_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    return {
        "config_name": config_name,
        "seed": int(seed),
        "log_dir": str(log_dir),
        "scores": scores.sort_values("path").reset_index(drop=True),
        "metrics": metrics,
        "run_config": run_config,
    }


def validate_paired_runs(runs):
    if not runs:
        raise ValueError("At least one run is required")
    reference = runs[0]
    reference_metadata = reference["scores"][METADATA_COLUMNS]
    manifest_hash = reference["run_config"]["manifest_sha256"]
    split_version = reference["run_config"]["split_version"]
    for run in runs[1:]:
        if not reference_metadata.equals(run["scores"][METADATA_COLUMNS]):
            raise ValueError(
                f"Test rows differ between {reference['log_dir']} and {run['log_dir']}"
            )
        if run["run_config"]["manifest_sha256"] != manifest_hash:
            raise ValueError("Runs use different manifests")
        if run["run_config"]["split_version"] != split_version:
            raise ValueError("Runs use different split versions")
    return manifest_hash, split_version


def make_cluster_plan(rows, rng):
    group_labels = rows.groupby("source_group")["label"].nunique()
    if (group_labels != 1).any():
        raise ValueError("Source groups must have one label")
    plan = []
    for label in (0, 1):
        groups = rows.loc[rows["label"] == label, "source_group"].drop_duplicates()
        sampled = rng.choice(groups.to_numpy(), size=len(groups), replace=True)
        plan.extend((label, draw_index, group) for draw_index, group in enumerate(sampled))
    return plan


def apply_cluster_plan(rows, plan):
    blocks = []
    for label, draw_index, source_group in plan:
        block = rows.loc[rows["source_group"] == source_group].copy()
        bootstrap_group = f"bootstrap_{label}_{draw_index}"
        block["source_group"] = bootstrap_group
        block["object_group"] = block["object_group"].map(
            lambda object_group: f"{bootstrap_group}:{object_group}"
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def primary_metric(rows):
    return multilevel_ranking_metrics(rows, "score")[PRIMARY_METRIC]


def compare_runs(
    baseline_runs,
    candidate_runs,
    iterations,
    bootstrap_seed,
    non_inferiority_margin,
):
    if not baseline_runs or not candidate_runs:
        raise ValueError("Baseline and candidate runs must not be empty")
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    if non_inferiority_margin < 0:
        raise ValueError("Non-inferiority margin must be non-negative")
    if [run["seed"] for run in baseline_runs] != [run["seed"] for run in candidate_runs]:
        raise ValueError("Baseline and candidate seeds do not match")
    validate_paired_runs(baseline_runs + candidate_runs)

    per_seed = []
    for baseline, candidate in zip(baseline_runs, candidate_runs, strict=True):
        baseline_metric = primary_metric(baseline["scores"])
        candidate_metric = primary_metric(candidate["scores"])
        per_seed.append(
            {
                "seed": baseline["seed"],
                "baseline": baseline_metric,
                "candidate": candidate_metric,
                "difference": candidate_metric - baseline_metric,
            }
        )

    rng = np.random.default_rng(bootstrap_seed)
    differences = []
    num_seeds = len(baseline_runs)
    reference_rows = baseline_runs[0]["scores"]
    for _ in range(iterations):
        sampled_seed_indices = rng.integers(0, num_seeds, size=num_seeds)
        cluster_plan = make_cluster_plan(reference_rows, rng)
        seed_differences = []
        for seed_index in sampled_seed_indices:
            baseline_rows = apply_cluster_plan(
                baseline_runs[int(seed_index)]["scores"],
                cluster_plan,
            )
            candidate_rows = apply_cluster_plan(
                candidate_runs[int(seed_index)]["scores"],
                cluster_plan,
            )
            seed_differences.append(
                primary_metric(candidate_rows) - primary_metric(baseline_rows)
            )
        differences.append(float(np.mean(seed_differences)))

    differences = np.asarray(differences, dtype=float)
    mean_difference = float(np.mean([row["difference"] for row in per_seed]))
    ci_low = float(np.quantile(differences, 0.025))
    ci_high = float(np.quantile(differences, 0.975))
    return {
        "per_seed": per_seed,
        "mean_difference": mean_difference,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "non_inferiority_margin": float(non_inferiority_margin),
        "non_inferior": bool(ci_low > -non_inferiority_margin),
        "bootstrap_probability_non_inferior": float(
            np.mean(differences > -non_inferiority_margin)
        ),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_method": (
            "paired hierarchical bootstrap: training seeds and label-stratified "
            "source-image groups"
        ),
    }


def seed_metric_row(run):
    metrics = run["metrics"]
    return {
        "config_name": run["config_name"],
        "seed": run["seed"],
        "selected_top_k_pixels": metrics["selected_top_k_pixels"],
        "selected_top_k_fraction": metrics["selected_top_k_fraction"],
        "primary_roc_auc": metrics[f"test_{PRIMARY_METRIC}"],
        "primary_roc_auc_ci_low": metrics["test_primary_roc_auc_ci_low"],
        "primary_roc_auc_ci_high": metrics["test_primary_roc_auc_ci_high"],
        "object_roc_auc": metrics["test_object_roc_auc"],
        "source_image_roc_auc": metrics["test_source_image_roc_auc"],
        "balanced_accuracy": metrics[
            "test_source_group_balanced_tile_balanced_accuracy"
        ],
        "fpr": metrics["test_source_group_balanced_tile_fpr"],
        "fnr": metrics["test_source_group_balanced_tile_fnr"],
        "log_dir": run["log_dir"],
    }


def write_outputs(output_dir, try_number, seed_rows, report, allow_overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fastflow_printer384_v2_final_try_{try_number}_comparison"
    csv_path = output_dir / f"{stem}_per_seed.csv"
    json_path = output_dir / f"{stem}.json"
    text_path = output_dir / f"{stem}.txt"
    for path in (csv_path, json_path, text_path):
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")

    pd.DataFrame(seed_rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"split_version: {report['split_version']}",
        f"primary_metric: {report['primary_metric']}",
    ]
    for baseline, comparison in report["comparisons"].items():
        lines.extend(
            [
                "",
                f"candidate_minus_{baseline}: {comparison['mean_difference']}",
                f"ci_95: [{comparison['bootstrap_ci_low']}, {comparison['bootstrap_ci_high']}]",
                f"margin: {comparison['non_inferiority_margin']}",
                f"non_inferior: {comparison['non_inferior']}",
            ]
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, text_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare printer FastFlow backbones.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments/printer"),
    )
    parser.add_argument("--try-number", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025])
    parser.add_argument(
        "--baseline-configs",
        nargs="+",
        default=["resnet18_256", "resnet18_384"],
    )
    parser.add_argument("--candidate-config", default="deit_base_distilled_384")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--non-inferiority-margin", type=float, default=0.03)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/printer/printer384_v2_final_summaries"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_names = [*args.baseline_configs, args.candidate_config]
    unknown = set(config_names) - set(PRINTER_CONFIGS)
    if unknown:
        raise KeyError(f"Unknown configs: {sorted(unknown)}")

    runs = {
        config_name: [
            load_run(args.experiments_root, config_name, args.try_number, seed)
            for seed in args.seeds
        ]
        for config_name in config_names
    }
    all_runs = [run for config_runs in runs.values() for run in config_runs]
    manifest_hash, split_version = validate_paired_runs(all_runs)

    comparisons = {
        baseline: compare_runs(
            runs[baseline],
            runs[args.candidate_config],
            iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
            non_inferiority_margin=args.non_inferiority_margin,
        )
        for baseline in args.baseline_configs
    }
    seed_rows = [seed_metric_row(run) for run in all_runs]
    report = {
        "split_version": split_version,
        "manifest_sha256": manifest_hash,
        "primary_metric": PRIMARY_METRIC,
        "candidate": args.candidate_config,
        "baselines": args.baseline_configs,
        "seeds": args.seeds,
        "comparisons": comparisons,
        "interpretation": (
            "Non-inferiority is supported only when the lower 95% bootstrap bound "
            "for candidate minus baseline is above the negative margin."
        ),
    }
    paths = write_outputs(
        args.output_dir,
        args.try_number,
        seed_rows,
        report,
        args.allow_overwrite,
    )
    print(json.dumps({"outputs": [str(path) for path in paths], **report}, indent=2))


if __name__ == "__main__":
    main()
