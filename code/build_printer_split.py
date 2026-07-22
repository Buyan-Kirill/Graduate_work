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


def records_by_group(records):
    result = defaultdict(list)
    for record in records:
        result[record["source_group"]].append(record)
    return result


def choose_normal_val_groups(records, fraction, seed):
    groups_by_date = defaultdict(set)
    for record in records:
        groups_by_date[record["source_date"]].add(record["source_group"])

    selected = set()
    for date, groups in groups_by_date.items():
        ordered = sorted(groups, key=lambda group: stable_key(seed, "normal-val", date, group))
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        selected.update(ordered[:count])
    return selected


def interleaved_records(records, seed, max_per_group=None):
    grouped = records_by_group(records)
    by_date = defaultdict(list)
    for source_group, group_records in grouped.items():
        ordered = sorted(
            group_records,
            key=lambda record: stable_key(seed, "path", record["path"]),
        )
        if max_per_group is not None:
            ordered = ordered[:max_per_group]
        by_date[ordered[0]["source_date"]].append((source_group, ordered))

    date_queues = {}
    for date, date_groups in by_date.items():
        date_groups.sort(key=lambda item: stable_key(seed, "group", item[0]))
        queue = []
        max_depth = max(len(group_records) for _, group_records in date_groups)
        for depth in range(max_depth):
            for _, group_records in date_groups:
                if depth < len(group_records):
                    queue.append(group_records[depth])
        date_queues[date] = queue

    result = []
    dates = sorted(date_queues, key=lambda date: stable_key(seed, "date", date))
    while any(date_queues.values()):
        for date in dates:
            if date_queues[date]:
                result.append(date_queues[date].pop(0))
    return result


def select_records(records, target, seed, max_per_group=None):
    ordered = interleaved_records(records, seed, max_per_group=max_per_group)
    if len(ordered) < target:
        raise ValueError(f"Requested {target} records, but only {len(ordered)} are available")
    return ordered[:target]


def manifest_row(record, dataset_root, split, class_name):
    return {
        "path": record["path"].resolve().relative_to(dataset_root).as_posix(),
        "split": split,
        "class_name": class_name,
        "label": record["label"],
        "source_date": record["source_date"],
        "source_group": record["source_group"],
        "sha256": record["sha256"],
    }


def selected_source_records(current_rows, source_index, split=None, label=None):
    records = []
    for row in current_rows:
        if split is not None and row["split"] != split:
            continue
        if label is not None and row["label"] != label:
            continue
        matches = source_index[row["sha256"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one canonical source for {row['path']}")
        records.append(matches[0])
    return records


def validate_manifest(rows):
    hashes_by_split = defaultdict(set)
    groups_by_split_class = defaultdict(set)
    duplicate_rows = 0
    for row in rows:
        if row["sha256"] in hashes_by_split[row["split"]]:
            duplicate_rows += 1
        hashes_by_split[row["split"]].add(row["sha256"])
        groups_by_split_class[(row["split"], row["class_name"])].add(row["source_group"])

    compared_pairs = (
        ("train", "normal_val_loss", "good"),
        ("calibration", "test", "good"),
        ("calibration", "test", "anomalies"),
    )
    group_overlap = {
        f"{left}_vs_{right}/{class_name}": sorted(
            groups_by_split_class[(left, class_name)]
            & groups_by_split_class[(right, class_name)]
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
    group_counts = {
        f"{split}/{class_name}": len(groups)
        for (split, class_name), groups in sorted(groups_by_split_class.items())
    }
    return {
        "counts": {
            f"{split}/{class_name}": count
            for (split, class_name), count in sorted(counts.items())
        },
        "source_group_counts": group_counts,
        "group_overlap": group_overlap,
        "exact_hash_overlap": hash_overlap,
        "duplicate_rows_within_split": duplicate_rows,
        "is_group_disjoint": not any(group_overlap.values()),
        "is_hash_disjoint": not any(hash_overlap.values()) and duplicate_rows == 0,
    }


def build_manifest(dataset_root, config):
    seed = int(config["seed"])
    fit_records = pool_records(dataset_root, FIT_TRAIN_DATES, label=0)
    heldout_records = pool_records(dataset_root, HELDOUT_NORMAL_DATES, label=0)
    anomaly_records = anomaly_pool_records(dataset_root)
    heldout_index = index_by_hash(heldout_records)
    anomaly_index = index_by_hash(anomaly_records)
    current_rows, unresolved = eval_manifest_rows(
        dataset_root,
        heldout_index,
        anomaly_index,
    )
    if unresolved:
        raise ValueError(f"Unresolved curated files: {unresolved}")

    normal_val_groups = choose_normal_val_groups(
        fit_records,
        float(config["normal_val_group_fraction"]),
        seed,
    )
    normal_val_candidates = [
        record for record in fit_records if record["source_group"] in normal_val_groups
    ]
    train_candidates = [
        record for record in fit_records if record["source_group"] not in normal_val_groups
    ]
    train_records = select_records(train_candidates, int(config["train_target"]), seed)
    normal_val_records = select_records(
        normal_val_candidates,
        int(config["normal_val_loss_target"]),
        seed + 1,
    )

    test_good_records = selected_source_records(
        current_rows,
        heldout_index,
        split="test",
        label=0,
    )
    test_good_groups = {record["source_group"] for record in test_good_records}
    calibration_good_candidates = [
        record for record in heldout_records if record["source_group"] not in test_good_groups
    ]
    calibration_good_records = select_records(
        calibration_good_candidates,
        int(config["calibration_good_target"]),
        seed + 2,
        max_per_group=int(config["calibration_good_max_per_group"]),
    )

    selected_anomaly_records = selected_source_records(current_rows, anomaly_index, label=1)
    calibration_anomaly_groups = set(config["calibration_anomaly_source_groups"])
    available_anomaly_groups = {
        record["source_group"] for record in selected_anomaly_records
    }
    missing_groups = calibration_anomaly_groups - available_anomaly_groups
    if missing_groups:
        raise ValueError(f"Calibration anomaly groups have no curated files: {sorted(missing_groups)}")

    rows = []
    for split, class_name, records in (
        ("train", "good", train_records),
        ("normal_val_loss", "good", normal_val_records),
        ("calibration", "good", calibration_good_records),
        ("test", "good", test_good_records),
    ):
        rows.extend(manifest_row(record, dataset_root, split, class_name) for record in records)

    for record in selected_anomaly_records:
        split = (
            "calibration"
            if record["source_group"] in calibration_anomaly_groups
            else "test"
        )
        rows.append(manifest_row(record, dataset_root, split, "anomalies"))

    rows.sort(key=lambda row: (row["split"], row["class_name"], row["path"]))
    return rows, validate_manifest(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a group-disjoint printer manifest.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/processed_printer_dataset_384"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/printer_split_v1.json"),
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

    dataset_root = args.dataset_root.resolve()
    rows, validation = build_manifest(dataset_root, config)
    report = {
        "version": config["version"],
        "approved": bool(config.get("approved")),
        "preview": not bool(config.get("approved")),
        **validation,
    }
    write_csv(args.manifest_output, rows)
    write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
