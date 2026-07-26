import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from printer_dataset_manifest import (
    FIT_TRAIN_DATES,
    HELDOUT_NORMAL_DATES,
    anomaly_pool_records,
    eval_manifest_rows,
    index_by_hash,
    pool_records,
    write_csv,
    write_json,
)


def stable_key(seed, *parts):
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def enrich_normal(record):
    prefix, date, object_stem = record["source_group"].split(":", maxsplit=2)
    if prefix != "normal" or "_obj" not in object_stem:
        raise ValueError(f"Cannot derive source image from {record['source_group']}")
    capture_stem = object_stem.rsplit("_obj", maxsplit=1)[0]
    return {
        **record,
        "capture_group": f"normal:{date}:{capture_stem}",
        "object_group": record["source_group"],
    }


def enrich_anomaly(record):
    return {
        **record,
        "capture_group": record["source_group"],
        "object_group": record["source_group"],
    }


def selected_source_records(current_rows, source_index, label):
    records = []
    for row in current_rows:
        if row["label"] != label:
            continue
        matches = source_index[row["sha256"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one canonical source for {row['path']}")
        records.append(matches[0])
    return records


def choose_normal_val_captures(records, fraction, seed):
    captures_by_date = defaultdict(set)
    for record in records:
        captures_by_date[record["source_date"]].add(record["capture_group"])

    selected = set()
    for date, captures in captures_by_date.items():
        ordered = sorted(
            captures,
            key=lambda capture: stable_key(seed, "normal-val", date, capture),
        )
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        selected.update(ordered[:count])
    return selected


def select_balanced(
    records,
    target,
    seed,
    stratum_key,
    item_group_key,
    max_per_item_group=None,
):
    by_stratum = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_stratum[record[stratum_key]][record[item_group_key]].append(record)

    queues = {}
    for stratum, item_groups in by_stratum.items():
        ordered_groups = sorted(
            item_groups,
            key=lambda group: stable_key(seed, "item-group", stratum, group),
        )
        ordered_records = {}
        for group in ordered_groups:
            values = sorted(
                item_groups[group],
                key=lambda record: stable_key(seed, "path", record["path"]),
            )
            if max_per_item_group is not None:
                values = values[:max_per_item_group]
            ordered_records[group] = values

        queue = []
        max_depth = max(len(values) for values in ordered_records.values())
        for depth in range(max_depth):
            for group in ordered_groups:
                values = ordered_records[group]
                if depth < len(values):
                    queue.append(values[depth])
        queues[stratum] = queue

    ordered_strata = sorted(
        queues,
        key=lambda stratum: stable_key(seed, "stratum", stratum),
    )
    selected = []
    while any(queues.values()) and len(selected) < target:
        for stratum in ordered_strata:
            if queues[stratum] and len(selected) < target:
                selected.append(queues[stratum].pop(0))

    if len(selected) != target:
        raise ValueError(f"Requested {target} records, but only {len(selected)} are available")
    return selected


def manifest_row(record, dataset_root, split, class_name):
    return {
        "path": record["path"].resolve().relative_to(dataset_root).as_posix(),
        "split": split,
        "class_name": class_name,
        "label": record["label"],
        "source_date": record["source_date"],
        "source_group": record["capture_group"],
        "object_group": record["object_group"],
        "sha256": record["sha256"],
    }


def validate_configured_groups(records, calibration_groups, test_groups, class_name):
    calibration_groups = set(calibration_groups)
    test_groups = set(test_groups)
    if calibration_groups & test_groups:
        raise ValueError(f"Overlapping configured {class_name} groups")
    available_groups = {record["capture_group"] for record in records}
    configured_groups = calibration_groups | test_groups
    if available_groups != configured_groups:
        missing = sorted(configured_groups - available_groups)
        unassigned = sorted(available_groups - configured_groups)
        raise ValueError(
            f"Configured {class_name} groups do not match data: "
            f"missing={missing}, unassigned={unassigned}"
        )


def validate_manifest(rows):
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest contains duplicate paths")

    hashes_by_split = defaultdict(set)
    captures_by_split_class = defaultdict(set)
    objects_by_split_class = defaultdict(set)
    for row in rows:
        hashes_by_split[row["split"]].add(row["sha256"])
        key = (row["split"], row["class_name"])
        captures_by_split_class[key].add(row["source_group"])
        objects_by_split_class[key].add(row["object_group"])

    compared_pairs = (
        ("train", "normal_val_loss", "good"),
        ("calibration", "test", "good"),
        ("calibration", "test", "anomalies"),
    )
    capture_overlap = {
        f"{left}_vs_{right}/{class_name}": sorted(
            captures_by_split_class[(left, class_name)]
            & captures_by_split_class[(right, class_name)]
        )
        for left, right, class_name in compared_pairs
    }
    object_overlap = {
        f"{left}_vs_{right}/{class_name}": sorted(
            objects_by_split_class[(left, class_name)]
            & objects_by_split_class[(right, class_name)]
        )
        for left, right, class_name in compared_pairs
    }

    split_names = sorted(hashes_by_split)
    hash_overlap = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            hash_overlap[f"{left}_vs_{right}"] = len(
                hashes_by_split[left] & hashes_by_split[right]
            )

    counts = Counter((row["split"], row["class_name"]) for row in rows)
    capture_counts = {
        f"{split}/{class_name}": len(groups)
        for (split, class_name), groups in sorted(captures_by_split_class.items())
    }
    object_counts = {
        f"{split}/{class_name}": len(groups)
        for (split, class_name), groups in sorted(objects_by_split_class.items())
    }
    return {
        "counts": {
            f"{split}/{class_name}": count
            for (split, class_name), count in sorted(counts.items())
        },
        "source_image_group_counts": capture_counts,
        "object_group_counts": object_counts,
        "source_image_group_overlap": capture_overlap,
        "object_group_overlap": object_overlap,
        "exact_hash_overlap": hash_overlap,
        "is_source_image_disjoint": not any(capture_overlap.values()),
        "is_object_disjoint": not any(object_overlap.values()),
        "is_hash_disjoint": not any(hash_overlap.values()),
    }


def build_manifest(dataset_root, config):
    seed = int(config["seed"])
    fit_records = [
        enrich_normal(record)
        for record in pool_records(dataset_root, FIT_TRAIN_DATES, label=0)
    ]
    heldout_records = [
        enrich_normal(record)
        for record in pool_records(dataset_root, HELDOUT_NORMAL_DATES, label=0)
    ]
    anomaly_records = [
        enrich_anomaly(record) for record in anomaly_pool_records(dataset_root)
    ]

    heldout_index = index_by_hash(heldout_records)
    anomaly_index = index_by_hash(anomaly_records)
    current_rows, unresolved = eval_manifest_rows(
        dataset_root,
        heldout_index,
        anomaly_index,
    )
    if unresolved:
        raise ValueError(f"Unresolved curated files: {unresolved}")

    normal_val_captures = choose_normal_val_captures(
        fit_records,
        float(config["normal_val_capture_fraction"]),
        seed,
    )
    train_candidates = [
        record for record in fit_records
        if record["capture_group"] not in normal_val_captures
    ]
    normal_val_candidates = [
        record for record in fit_records
        if record["capture_group"] in normal_val_captures
    ]
    train_sampling_ranks = None
    if config.get("include_train_sampling_ranks"):
        date_balanced = select_balanced(
            train_candidates,
            len(train_candidates),
            seed,
            stratum_key="source_date",
            item_group_key="object_group",
        )
        object_uniform_candidates = [
            {**record, "sampling_stratum": "all_dates"}
            for record in train_candidates
        ]
        object_uniform = select_balanced(
            object_uniform_candidates,
            len(object_uniform_candidates),
            seed,
            stratum_key="sampling_stratum",
            item_group_key="object_group",
        )

        def rank_by_path(ordered_records):
            return {
                record["path"].resolve().relative_to(dataset_root).as_posix(): rank
                for rank, record in enumerate(ordered_records, start=1)
            }

        train_sampling_ranks = {
            "train_rank_date_balanced": rank_by_path(date_balanced),
            "train_rank_object_uniform": rank_by_path(object_uniform),
        }

    train_records = select_balanced(
        train_candidates,
        int(config["train_target"]),
        seed,
        stratum_key="source_date",
        item_group_key="object_group",
    )
    normal_val_records = select_balanced(
        normal_val_candidates,
        int(config["normal_val_loss_target"]),
        seed + 1,
        stratum_key="source_date",
        item_group_key="object_group",
        max_per_item_group=int(config["normal_val_loss_max_per_object"]),
    )

    calibration_normal_groups = set(config["calibration_normal_capture_groups"])
    test_normal_groups = set(config["test_normal_capture_groups"])
    validate_configured_groups(
        heldout_records,
        calibration_normal_groups,
        test_normal_groups,
        "normal",
    )
    calibration_good_records = select_balanced(
        [record for record in heldout_records if record["capture_group"] in calibration_normal_groups],
        int(config["calibration_good_target"]),
        seed + 2,
        stratum_key="capture_group",
        item_group_key="object_group",
        max_per_item_group=int(config["max_eval_images_per_object"]),
    )
    test_good_records = select_balanced(
        [record for record in heldout_records if record["capture_group"] in test_normal_groups],
        int(config["test_good_target"]),
        seed + 3,
        stratum_key="capture_group",
        item_group_key="object_group",
        max_per_item_group=int(config["max_eval_images_per_object"]),
    )

    excluded_object_groups = set(config.get("excluded_eval_object_groups", {}))
    selected_object_groups = {
        record["object_group"]
        for record in calibration_good_records + test_good_records
    }
    if not excluded_object_groups <= selected_object_groups:
        raise ValueError(
            "Excluded object groups are not in the selected evaluation data: "
            f"{sorted(excluded_object_groups - selected_object_groups)}"
        )
    calibration_good_records = [
        record
        for record in calibration_good_records
        if record["object_group"] not in excluded_object_groups
    ]
    test_good_records = [
        record
        for record in test_good_records
        if record["object_group"] not in excluded_object_groups
    ]

    selected_anomalies = selected_source_records(current_rows, anomaly_index, label=1)
    selected_paths = {
        record["path"].resolve().relative_to(dataset_root).as_posix()
        for record in selected_anomalies
    }
    excluded_paths = set(config.get("excluded_eval_paths", {}))
    if not excluded_paths <= selected_paths:
        raise ValueError(f"Excluded paths are not in curated evaluation data: {sorted(excluded_paths - selected_paths)}")
    selected_anomalies = [
        record
        for record in selected_anomalies
        if record["path"].resolve().relative_to(dataset_root).as_posix() not in excluded_paths
    ]

    calibration_anomaly_groups = set(config["calibration_anomaly_source_groups"])
    test_anomaly_groups = set(config["test_anomaly_source_groups"])
    validate_configured_groups(
        selected_anomalies,
        calibration_anomaly_groups,
        test_anomaly_groups,
        "anomaly",
    )

    rows = []
    for split, class_name, records in (
        ("train", "good", train_records),
        ("normal_val_loss", "good", normal_val_records),
        ("calibration", "good", calibration_good_records),
        ("test", "good", test_good_records),
        (
            "calibration",
            "anomalies",
            [record for record in selected_anomalies if record["capture_group"] in calibration_anomaly_groups],
        ),
        (
            "test",
            "anomalies",
            [record for record in selected_anomalies if record["capture_group"] in test_anomaly_groups],
        ),
    ):
        rows.extend(manifest_row(record, dataset_root, split, class_name) for record in records)

    rows.sort(key=lambda row: (row["split"], row["class_name"], row["path"]))
    validation = validate_manifest(rows)
    if train_sampling_ranks is not None:
        for row in rows:
            for column, ranks in train_sampling_ranks.items():
                row[column] = ranks.get(row["path"]) if row["split"] == "train" else None
        validation["train_sampling_ranks"] = {
            column: {
                "ranked_records": len(ranks),
                "min_rank": min(ranks.values()),
                "max_rank": max(ranks.values()),
            }
            for column, ranks in train_sampling_ranks.items()
        }
    return rows, validation


def parse_args():
    parser = argparse.ArgumentParser(description="Build printer split v2 by source image.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/processed_printer_dataset_384"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/printer_split_v2.json"),
    )
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--allow-unapproved-preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config.get("approved") and not args.allow_unapproved_preview:
        raise RuntimeError(
            "Split config is not approved. Use --allow-unapproved-preview only for review output."
        )

    for output in (args.manifest_output, args.report_output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")

    dataset_root = args.dataset_root.resolve()
    rows, validation = build_manifest(dataset_root, config)
    report = {
        "version": config["version"],
        "approved": bool(config.get("approved")),
        "preview": not bool(config.get("approved")),
        "grouping": {
            "normal_split_unit": "source image",
            "normal_object_unit": "instance-mask object",
            "anomaly_split_unit": "source capture",
        },
        "excluded_eval_paths": config.get("excluded_eval_paths", {}),
        "excluded_eval_object_groups": config.get("excluded_eval_object_groups", {}),
        **validation,
    }
    write_csv(args.manifest_output, rows)
    write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
