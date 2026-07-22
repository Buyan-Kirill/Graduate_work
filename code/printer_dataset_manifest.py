import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIT_TRAIN_DATES = (
    "2024-12-16",
    "2025-01-15",
    "2025-01-29",
    "2025-04-02",
    "2025-05-26",
    "2025-05-28",
)
HELDOUT_NORMAL_DATES = ("2025-02-07_test", "2025-06-25_test")


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tile_source_stem(path):
    stem = Path(path).stem
    stem = re.sub(r"_tile_\d+_\d+$", "", stem)
    return re.sub(r"_centered$", "", stem)


def relative_posix(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def image_metadata(path):
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image.verify()
    return width, height, mode


def pool_records(dataset_root, dates, label):
    records = []
    for date in dates:
        folder = dataset_root / "training" / date / "objects_parts"
        for path in list_images(folder):
            records.append(
                {
                    "path": path,
                    "sha256": sha256(path),
                    "source_date": date,
                    "source_group": f"normal:{date}:{tile_source_stem(path)}",
                    "label": label,
                }
            )
    return records


def anomaly_pool_records(dataset_root):
    records = []
    for path in list_images(dataset_root / "anomalies" / "objects_parts"):
        records.append(
            {
                "path": path,
                "sha256": sha256(path),
                "source_date": tile_source_stem(path).split("_L", maxsplit=1)[0],
                "source_group": f"anomaly:{tile_source_stem(path)}",
                "label": 1,
            }
        )
    return records


def index_by_hash(records):
    result = {}
    for record in records:
        result.setdefault(record["sha256"], []).append(record)
    return result


def eval_manifest_rows(dataset_root, normal_index, anomaly_index):
    rows = []
    unresolved = []
    for split in ("val", "test"):
        for class_name, label, source_index in (
            ("good", 0, normal_index),
            ("anomalies", 1, anomaly_index),
        ):
            folder = dataset_root / split / class_name
            for path in list_images(folder):
                digest = sha256(path)
                matches = source_index.get(digest, [])
                if len(matches) != 1:
                    unresolved.append(
                        {
                            "path": relative_posix(path, dataset_root),
                            "match_count": len(matches),
                        }
                    )
                    continue
                source = matches[0]
                width, height, mode = image_metadata(path)
                rows.append(
                    {
                        "path": relative_posix(path, dataset_root),
                        "canonical_path": relative_posix(source["path"], dataset_root),
                        "split": split,
                        "class_name": class_name,
                        "label": label,
                        "source_date": source["source_date"],
                        "source_group": source["source_group"],
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "mode": mode,
                    }
                )
    return rows, unresolved


def build_report(dataset_root, rows, unresolved, fit_train_records):
    by_split_class = Counter((row["split"], row["class_name"]) for row in rows)
    groups = {
        (split, label): {
            row["source_group"]
            for row in rows
            if row["split"] == split and row["label"] == label
        }
        for split in ("val", "test")
        for label in (0, 1)
    }
    hashes = {
        split: {row["sha256"] for row in rows if row["split"] == split}
        for split in ("val", "test")
    }
    fit_train_hashes = {record["sha256"] for record in fit_train_records}
    eval_hashes = hashes["val"] | hashes["test"]
    duplicate_hashes = {
        split: sum(row["split"] == split for row in rows) - len(hashes[split])
        for split in ("val", "test")
    }
    unexpected_image_specs = [
        row["path"]
        for row in rows
        if (row["width"], row["height"], row["mode"]) != (384, 384, "RGB")
    ]

    shared_groups = {
        "good": sorted(groups[("val", 0)] & groups[("test", 0)]),
        "anomalies": sorted(groups[("val", 1)] & groups[("test", 1)]),
    }
    affected_val = {
        class_name: sum(
            row["split"] == "val"
            and row["class_name"] == class_name
            and row["source_group"] in set(shared_groups[class_name])
            for row in rows
        )
        for class_name in ("good", "anomalies")
    }

    try:
        report_root = Path(dataset_root).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        report_root = str(Path(dataset_root).resolve())

    is_group_disjoint = not shared_groups["good"] and not shared_groups["anomalies"]
    return {
        "dataset_root": report_root,
        "counts": {
            f"{split}/{class_name}": by_split_class[(split, class_name)]
            for split in ("val", "test")
            for class_name in ("good", "anomalies")
        },
        "source_group_counts": {
            f"{split}/{class_name}": len(groups[(split, int(class_name == "anomalies"))])
            for split in ("val", "test")
            for class_name in ("good", "anomalies")
        },
        "exact_hash_overlap": {
            "val_vs_test": len(hashes["val"] & hashes["test"]),
            "fit_train_vs_eval": len(fit_train_hashes & eval_hashes),
        },
        "duplicate_hashes_within_split": duplicate_hashes,
        "shared_val_test_source_groups": shared_groups,
        "validation_files_from_shared_source_groups": affected_val,
        "unexpected_image_specs": unexpected_image_specs,
        "unresolved_files": unresolved,
        "is_valid_locked_test_split": (
            is_group_disjoint
            and not (hashes["val"] & hashes["test"])
            and not (fit_train_hashes & eval_hashes)
            and not unexpected_image_specs
            and not unresolved
        ),
    }


def build_group_summary(pool_records, manifest_rows):
    pool_counts = Counter(record["source_group"] for record in pool_records)
    split_counts = Counter(
        (row["source_group"], row["split"]) for row in manifest_rows
    )
    metadata = {
        record["source_group"]: {
            "class_name": "anomalies" if record["label"] == 1 else "good",
            "source_date": record["source_date"],
        }
        for record in pool_records
    }
    summary = []
    for source_group in sorted(pool_counts):
        val_count = split_counts[(source_group, "val")]
        test_count = split_counts[(source_group, "test")]
        summary.append(
            {
                "source_group": source_group,
                **metadata[source_group],
                "pool_files": pool_counts[source_group],
                "current_val_files": val_count,
                "current_test_files": test_count,
                "current_unused_files": pool_counts[source_group] - val_count - test_count,
                "spans_val_and_test": bool(val_count and test_count),
            }
        )
    return summary


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit the curated printer val/test split.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/processed_printer_dataset_384"),
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--group-summary-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    fit_train_records = pool_records(dataset_root, FIT_TRAIN_DATES, label=0)
    heldout_normal_records = pool_records(dataset_root, HELDOUT_NORMAL_DATES, label=0)
    anomaly_records = anomaly_pool_records(dataset_root)
    rows, unresolved = eval_manifest_rows(
        dataset_root,
        index_by_hash(heldout_normal_records),
        index_by_hash(anomaly_records),
    )
    report = build_report(dataset_root, rows, unresolved, fit_train_records)
    group_summary = build_group_summary(
        heldout_normal_records + anomaly_records,
        rows,
    )

    if args.manifest_output:
        write_csv(args.manifest_output, rows)
    if args.group_summary_output:
        write_csv(args.group_summary_output, group_summary)
    if args.report_output:
        write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
