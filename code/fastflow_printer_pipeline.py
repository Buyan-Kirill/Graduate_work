import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from anomalib.models.image.fastflow.loss import FastflowLoss
from anomalib.models.image.fastflow.torch_model import FastflowModel


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TOP_K_FRACTIONS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.0625,
    0.10,
    0.20,
    1.0 / 3.0,
    0.50,
    1.0,
)
PRINTER_CONFIGS = {
    "resnet18_256": {
        "name": "resnet18_256",
        "tag": "fastflow_resnet18_256_printer384_v2_final",
        "backbone": "resnet18",
        "image_size": 256,
        "flow_steps": 10,
        "hidden_ratio": None,
        "num_epochs": 14,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "eta_min": 1e-6,
        "batch_size": 8,
        "grad_clip_norm": None,
        "augmentation": "none",
        "threshold_quantile": 0.95,
        "num_workers": 0,
    },
    "resnet18_384": {
        "name": "resnet18_384",
        "tag": "fastflow_resnet18_384_printer384_v2_final",
        "backbone": "resnet18",
        "image_size": 384,
        "flow_steps": 10,
        "hidden_ratio": None,
        "num_epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "eta_min": 1e-6,
        "batch_size": 8,
        "grad_clip_norm": None,
        "augmentation": "none",
        "threshold_quantile": 0.95,
        "num_workers": 0,
    },
    "deit_base_distilled_384": {
        "name": "deit_base_distilled_384",
        "tag": "fastflow_deit_base_distilled_384_printer384_v2_final",
        "backbone": "deit_base_distilled_patch16_384",
        "image_size": 384,
        "flow_steps": 8,
        "hidden_ratio": 0.5,
        "num_epochs": 40,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "eta_min": 1e-6,
        "batch_size": 10,
        "grad_clip_norm": 10.0,
        "augmentation": "none",
        "threshold_quantile": 0.95,
        "num_workers": 0,
    },
}
CONFIG_ROLES = {
    "resnet18_256": "historical printer baseline at 256 input",
    "resnet18_384": "resolution-matched ResNet18 control for DeiT-384",
    "deit_base_distilled_384": "transformer candidate using the MVTec-tested recipe",
}


def resolve_project_root(start=None):
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, current.parent):
        if (candidate / "datasets").exists() and (candidate / "experiments").exists():
            return candidate
    raise FileNotFoundError("Cannot resolve project root with datasets/ and experiments/")


def seed_everything(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def save_text_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for name, value in metrics.items():
            file.write(f"{name}: {value}\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(project_root):
    project_root = Path(project_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_dirty": None}
    return {
        "git_commit": commit,
        "git_worktree_dirty": bool(status.strip()),
    }


def write_run_note(path, run_config):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite run note: {path}")
    config = run_config["config"]
    lines = [
        f"# {config['name']} / seed {run_config['seed']}",
        "",
        f"Status at creation: planned train + calibration; test locked.",
        f"Role: {CONFIG_ROLES[config['name']]}.",
        f"Split: {run_config['split_version']}.",
        f"Manifest SHA-256: `{run_config['manifest_sha256']}`.",
        f"Pipeline SHA-256: `{run_config['pipeline_sha256']}`.",
        f"Git commit: `{run_config['git_commit']}`.",
        f"Git worktree dirty: `{run_config['git_worktree_dirty']}`.",
        "",
        "Training parameters:",
        "",
        f"- input: {config['image_size']}x{config['image_size']}",
        f"- epochs: {config['num_epochs']}",
        f"- flow_steps: {config['flow_steps']}",
        f"- learning_rate: {config['learning_rate']}",
        f"- weight_decay: {config['weight_decay']}",
        f"- batch_size: {config['batch_size']}",
        f"- grad_clip_norm: {config['grad_clip_norm']}",
        f"- augmentation: {config['augmentation']}",
        "",
        "Top-k and threshold are selected only on calibration. Test must not be",
        "read until the candidate configurations are frozen.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_approved_manifest(
    dataset_root,
    manifest_path,
    split_config_path,
    verify_hashes=True,
):
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    split_config = json.loads(Path(split_config_path).read_text(encoding="utf-8"))
    if not split_config.get("approved"):
        raise RuntimeError("Printer split is not approved; refusing to train or evaluate")

    manifest = pd.read_csv(manifest_path)
    required_columns = {
        "path",
        "split",
        "class_name",
        "label",
        "source_date",
        "source_group",
        "object_group",
        "sha256",
    }
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    expected_splits = {"train", "normal_val_loss", "calibration", "test"}
    if set(manifest["split"]) != expected_splits:
        raise ValueError(f"Expected splits {sorted(expected_splits)}, got {sorted(set(manifest['split']))}")
    if manifest["path"].duplicated().any():
        raise ValueError("Manifest contains duplicate paths")
    expected_labels = manifest["class_name"].map({"good": 0, "anomalies": 1})
    if expected_labels.isna().any() or not np.array_equal(
        expected_labels.to_numpy(dtype=int),
        manifest["label"].to_numpy(dtype=int),
    ):
        raise ValueError("Manifest class_name/label mapping is invalid")
    missing_files = [path for path in manifest["path"] if not (dataset_root / path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Manifest files are missing: {missing_files[:5]}")
    if verify_hashes:
        changed_files = [
            path
            for path, expected_hash in manifest[["path", "sha256"]].itertuples(index=False)
            if file_sha256(dataset_root / path) != expected_hash
        ]
        if changed_files:
            raise ValueError(f"Manifest hashes do not match files: {changed_files[:5]}")

    for class_name in ("good", "anomalies"):
        calibration_groups = set(
            manifest.loc[
                (manifest["split"] == "calibration")
                & (manifest["class_name"] == class_name),
                "source_group",
            ]
        )
        test_groups = set(
            manifest.loc[
                (manifest["split"] == "test")
                & (manifest["class_name"] == class_name),
                "source_group",
            ]
        )
        overlap = calibration_groups & test_groups
        if overlap:
            raise ValueError(f"Calibration/test source overlap for {class_name}: {sorted(overlap)}")

    train_groups = set(manifest.loc[manifest["split"] == "train", "source_group"])
    normal_val_groups = set(
        manifest.loc[manifest["split"] == "normal_val_loss", "source_group"]
    )
    if train_groups & normal_val_groups:
        raise ValueError("Train/normal_val_loss source groups overlap")

    train_objects = set(manifest.loc[manifest["split"] == "train", "object_group"])
    normal_val_objects = set(
        manifest.loc[manifest["split"] == "normal_val_loss", "object_group"]
    )
    if train_objects & normal_val_objects:
        raise ValueError("Train/normal_val_loss object groups overlap")

    split_hashes = {
        split: set(manifest.loc[manifest["split"] == split, "sha256"])
        for split in expected_splits
    }
    split_names = sorted(split_hashes)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if split_hashes[left] & split_hashes[right]:
                raise ValueError(f"Exact hash overlap between {left} and {right}")
    return manifest, split_config


def image_transform(image_size, augmentation="none"):
    steps = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]
    if augmentation == "mild_photometric":
        steps.extend(
            [
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8))],
                    p=0.15,
                ),
            ]
        )
    elif augmentation != "none":
        raise ValueError(f"Unknown augmentation: {augmentation}")
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class ManifestDataset(Dataset):
    def __init__(self, dataset_root, rows, transform, include_metadata=False):
        self.dataset_root = Path(dataset_root)
        self.rows = rows.reset_index(drop=True).copy()
        self.transform = transform
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.dataset_root / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        if not self.include_metadata:
            return tensor
        return {
            "image": tensor,
            "label": int(row["label"]),
            "path": row["path"],
            "source_group": row["source_group"],
            "object_group": row["object_group"],
            "source_date": row["source_date"],
        }


def make_loader(dataset, batch_size, shuffle, seed, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def build_fastflow_model(config):
    kwargs = {
        "backbone": config["backbone"],
        "flow_steps": config["flow_steps"],
        "input_size": (config["image_size"], config["image_size"]),
        "pre_trained": True,
    }
    if config.get("hidden_ratio") is not None:
        kwargs["hidden_ratio"] = config["hidden_ratio"]
    return FastflowModel(**kwargs)


def freeze_feature_extractor(model):
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False


def trainable_parameters(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def compute_fastflow_loss(criterion, outputs):
    hidden_variables = getattr(outputs, "hidden_variables", None)
    jacobians = getattr(outputs, "jacobians", None)
    if isinstance(outputs, dict):
        hidden_variables = outputs.get("hidden_variables", hidden_variables)
        jacobians = outputs.get("jacobians", jacobians)
    if (hidden_variables is None or jacobians is None) and isinstance(outputs, (tuple, list)):
        if len(outputs) >= 2:
            hidden_variables, jacobians = outputs[0], outputs[1]
    if hidden_variables is None or jacobians is None:
        raise TypeError("FastFlow training output does not contain hidden variables and jacobians")
    return criterion(hidden_variables, jacobians)


def load_model_weights(model, weights_path, device):
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def train_fastflow(model, train_dataset, normal_val_dataset, config, log_dir, seed, device):
    training_started = time.perf_counter()
    model = model.to(device)
    freeze_feature_extractor(model)
    parameters = trainable_parameters(model)
    criterion = FastflowLoss()
    optimizer = optim.AdamW(
        parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    train_loader = make_loader(
        train_dataset,
        config["batch_size"],
        shuffle=True,
        seed=seed,
        num_workers=config.get("num_workers", 0),
    )
    val_loader = make_loader(
        normal_val_dataset,
        config["batch_size"],
        shuffle=False,
        seed=seed,
        num_workers=config.get("num_workers", 0),
    )

    total_steps = max(1, config["num_epochs"] * len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=config.get("eta_min", 1e-6),
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    best_weights_path = Path(log_dir) / "best_model_weights.pth"
    for epoch in range(config["num_epochs"]):
        epoch_started = time.perf_counter()
        model.train()
        train_losses = []
        clipped_gradient_norms = []
        for images in tqdm(train_loader, desc=f"train {epoch + 1}/{config['num_epochs']}", leave=False):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_fastflow_loss(criterion, model(images))
            loss.backward()
            if config.get("grad_clip_norm") is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    config["grad_clip_norm"],
                )
                clipped_gradient_norms.append(float(gradient_norm.detach().cpu()))
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        # FastflowModel returns likelihood tensors only in train mode.
        model.train()
        val_losses = []
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                loss = compute_fastflow_loss(criterion, model(images))
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "normal_val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_duration_seconds": float(time.perf_counter() - epoch_started),
        }
        if clipped_gradient_norms:
            clip_norm = float(config["grad_clip_norm"])
            epoch_row.update(
                {
                    "gradient_norm_mean_before_clip": float(np.mean(clipped_gradient_norms)),
                    "gradient_norm_max_before_clip": float(np.max(clipped_gradient_norms)),
                    "gradient_clip_fraction": float(
                        np.mean(np.asarray(clipped_gradient_norms) > clip_norm)
                    ),
                }
            )
        history.append(epoch_row)
        print(
            f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
            f"normal_val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_weights_path)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": config,
                "seed": seed,
            },
            Path(log_dir) / "last_checkpoint.pth",
        )

    load_model_weights(model, best_weights_path, device)
    pd.DataFrame(history).to_csv(Path(log_dir) / "train_history.csv", index=False)
    save_json(
        Path(log_dir) / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_normal_val_loss": best_val_loss,
            "final_train_loss": history[-1]["train_loss"],
            "final_normal_val_loss": history[-1]["normal_val_loss"],
            "training_duration_seconds": float(time.perf_counter() - training_started),
        },
    )
    return model


def top_k_grid(num_pixels):
    candidates = {1, num_pixels}
    candidates.update(int(round(num_pixels * fraction)) for fraction in TOP_K_FRACTIONS)
    return sorted({max(1, min(num_pixels, value)) for value in candidates})


def collect_scores(model, dataset, top_k_values, device, batch_size, seed, desc):
    loader = make_loader(dataset, batch_size, shuffle=False, seed=seed)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            outputs = model(batch["image"].to(device))
            anomaly_maps = outputs.anomaly_map.detach().cpu().float()
            for index in range(len(batch["label"])):
                values = anomaly_maps[index].flatten().sort(descending=True).values
                cumulative = values.cumsum(dim=0)
                row = {
                    "path": batch["path"][index],
                    "source_group": batch["source_group"][index],
                    "object_group": batch["object_group"][index],
                    "source_date": batch["source_date"][index],
                    "label": int(batch["label"][index]),
                }
                for top_k in top_k_values:
                    row[f"score_top_{top_k}"] = float(cumulative[top_k - 1] / top_k)
                rows.append(row)
    return pd.DataFrame(rows)


def group_balanced_weights(source_groups):
    groups = pd.Series(source_groups)
    counts = groups.value_counts()
    return groups.map(lambda group: 1.0 / counts[group]).to_numpy(dtype=float)


def cohens_d(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    if len(normal) < 2 or len(anomaly) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(normal) - 1) * normal.var(ddof=1) + (len(anomaly) - 1) * anomaly.var(ddof=1))
        / (len(normal) + len(anomaly) - 2)
    )
    return float((anomaly.mean() - normal.mean()) / pooled) if pooled else float("nan")


def _ranking_values(labels, scores, sample_weight=None):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Ranking metrics require both normal and anomaly labels")
    return (
        float(roc_auc_score(labels, scores, sample_weight=sample_weight)),
        float(average_precision_score(labels, scores, sample_weight=sample_weight)),
    )


def aggregate_score_rows(rows, group_column, score_column):
    label_counts = rows.groupby(group_column, sort=False)["label"].nunique()
    mixed_groups = label_counts[label_counts != 1]
    if not mixed_groups.empty:
        raise ValueError(f"Groups contain mixed labels: {mixed_groups.index.tolist()[:5]}")
    return (
        rows.groupby(group_column, sort=False)
        .agg(label=("label", "first"), score=(score_column, "max"))
        .reset_index()
    )


def multilevel_ranking_metrics(rows, score_column):
    labels = rows["label"].to_numpy(dtype=int)
    scores = rows[score_column].to_numpy(dtype=float)
    source_weights = group_balanced_weights(rows["source_group"])
    tile_roc_auc, tile_average_precision = _ranking_values(labels, scores)
    weighted_roc_auc, weighted_average_precision = _ranking_values(
        labels,
        scores,
        sample_weight=source_weights,
    )

    object_rows = aggregate_score_rows(rows, "object_group", score_column)
    object_roc_auc, object_average_precision = _ranking_values(
        object_rows["label"],
        object_rows["score"],
    )
    source_rows = aggregate_score_rows(rows, "source_group", score_column)
    source_roc_auc, source_average_precision = _ranking_values(
        source_rows["label"],
        source_rows["score"],
    )
    return {
        "tile_roc_auc": tile_roc_auc,
        "tile_average_precision": tile_average_precision,
        "source_group_balanced_tile_roc_auc": weighted_roc_auc,
        "source_group_balanced_tile_average_precision": weighted_average_precision,
        "tile_cohens_d": cohens_d(labels, scores),
        "object_roc_auc": object_roc_auc,
        "object_average_precision": object_average_precision,
        "source_image_roc_auc": source_roc_auc,
        "source_image_average_precision": source_average_precision,
    }


def _threshold_values(labels, scores, threshold, sample_weight=None):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = scores >= threshold
    normal = labels == 0
    anomaly = labels == 1
    if not normal.any() or not anomaly.any():
        raise ValueError("Threshold metrics require both normal and anomaly labels")
    if sample_weight is None:
        sample_weight = np.ones(len(labels), dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    fpr = float(np.average(predictions[normal], weights=sample_weight[normal]))
    fnr = float(np.average(~predictions[anomaly], weights=sample_weight[anomaly]))
    return {
        "balanced_accuracy": float(1.0 - 0.5 * (fpr + fnr)),
        "fpr": fpr,
        "fnr": fnr,
    }


def multilevel_threshold_metrics(rows, score_column, threshold):
    labels = rows["label"].to_numpy(dtype=int)
    scores = rows[score_column].to_numpy(dtype=float)
    tile_metrics = _threshold_values(labels, scores, threshold)
    weighted_tile_metrics = _threshold_values(
        labels,
        scores,
        threshold,
        sample_weight=group_balanced_weights(rows["source_group"]),
    )
    object_rows = aggregate_score_rows(rows, "object_group", score_column)
    object_metrics = _threshold_values(
        object_rows["label"],
        object_rows["score"],
        threshold,
    )
    source_rows = aggregate_score_rows(rows, "source_group", score_column)
    source_metrics = _threshold_values(
        source_rows["label"],
        source_rows["score"],
        threshold,
    )
    return {
        "threshold": float(threshold),
        **{f"tile_{name}": value for name, value in tile_metrics.items()},
        **{
            f"source_group_balanced_tile_{name}": value
            for name, value in weighted_tile_metrics.items()
        },
        **{f"object_{name}": value for name, value in object_metrics.items()},
        **{f"source_image_{name}": value for name, value in source_metrics.items()},
    }


def select_top_k(calibration_scores, top_k_values, num_pixels):
    metric_rows = []
    for top_k in top_k_values:
        metrics = multilevel_ranking_metrics(calibration_scores, f"score_top_{top_k}")
        metric_rows.append(
            {
                "top_k_pixels": top_k,
                "top_k_fraction": float(top_k / num_pixels),
                **metrics,
            }
        )
    sweep = pd.DataFrame(metric_rows)
    ranked = sweep.sort_values(
        [
            "source_group_balanced_tile_roc_auc",
            "object_roc_auc",
            "tile_roc_auc",
            "source_group_balanced_tile_average_precision",
            "object_average_precision",
            "tile_average_precision",
            "top_k_pixels",
        ],
        ascending=[False, False, False, False, False, False, True],
    )
    selected = int(ranked.iloc[0]["top_k_pixels"])
    sweep["selected"] = sweep["top_k_pixels"] == selected
    return selected, sweep


def stratified_source_group_resample(rows, rng):
    group_labels = rows.groupby("source_group")["label"].nunique()
    if (group_labels != 1).any():
        raise ValueError("Source groups must have one label for cluster bootstrap")

    sampled_blocks = []
    for label in (0, 1):
        groups = rows.loc[rows["label"] == label, "source_group"].drop_duplicates().to_numpy()
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        for draw_index, source_group in enumerate(sampled_groups):
            block = rows.loc[rows["source_group"] == source_group].copy()
            prefix = f"bootstrap_{label}_{draw_index}"
            block["source_group"] = prefix
            block["object_group"] = block["object_group"].map(
                lambda object_group: f"{prefix}:{object_group}"
            )
            sampled_blocks.append(block)
    return pd.concat(sampled_blocks, ignore_index=True)


def bootstrap_confidence_intervals(rows, threshold, iterations, seed):
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    ranking = multilevel_ranking_metrics(rows, "score")
    thresholded = multilevel_threshold_metrics(rows, "score", threshold)
    metric_names = [
        "tile_roc_auc",
        "source_group_balanced_tile_roc_auc",
        "object_roc_auc",
        "source_image_roc_auc",
        "tile_balanced_accuracy",
        "tile_fpr",
        "tile_fnr",
        "source_group_balanced_tile_balanced_accuracy",
        "source_group_balanced_tile_fpr",
        "source_group_balanced_tile_fnr",
        "object_balanced_accuracy",
        "object_fpr",
        "object_fnr",
        "source_image_balanced_accuracy",
        "source_image_fpr",
        "source_image_fnr",
    ]
    point_estimates = {**ranking, **thresholded}
    distributions = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sampled = stratified_source_group_resample(rows, rng)
        sampled_metrics = {
            **multilevel_ranking_metrics(sampled, "score"),
            **multilevel_threshold_metrics(sampled, "score", threshold),
        }
        for name in metric_names:
            distributions[name].append(sampled_metrics[name])

    return {
        "method": "stratified source-group cluster bootstrap",
        "iterations": int(iterations),
        "seed": int(seed),
        "confidence_level": 0.95,
        "metrics": {
            name: {
                "estimate": float(point_estimates[name]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
            for name, values in distributions.items()
        },
    }


def calibrate_fastflow(
    model,
    calibration_dataset,
    config,
    log_dir,
    seed,
    device,
):
    num_pixels = config["image_size"] ** 2
    candidates = top_k_grid(num_pixels)
    calibration_all_scores = collect_scores(
        model,
        calibration_dataset,
        candidates,
        device,
        config["batch_size"],
        seed,
        "calibration top-k",
    )
    selected_top_k, sweep = select_top_k(calibration_all_scores, candidates, num_pixels)
    score_column = f"score_top_{selected_top_k}"
    sweep.to_csv(Path(log_dir) / "calibration_top_k_sweep.csv", index=False)
    calibration_all_scores.to_csv(
        Path(log_dir) / "calibration_top_k_scores.csv",
        index=False,
    )

    calibration_scores = calibration_all_scores[
        ["path", "source_group", "object_group", "source_date", "label", score_column]
    ].rename(columns={score_column: "score"})
    calibration_scores.to_csv(Path(log_dir) / "calibration_scores.csv", index=False)
    normal_scores = calibration_scores.loc[calibration_scores["label"] == 0, "score"]
    if normal_scores.empty:
        raise ValueError("Calibration requires normal samples for threshold selection")
    threshold_quantile = float(config.get("threshold_quantile", 0.95))
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between 0 and 1")
    threshold = float(np.quantile(normal_scores, threshold_quantile))

    calibration_ranking = multilevel_ranking_metrics(calibration_scores, "score")
    calibration_threshold = multilevel_threshold_metrics(
        calibration_scores,
        "score",
        threshold,
    )
    selection = {
        "image_size": int(config["image_size"]),
        "num_pixels": int(num_pixels),
        "selected_top_k_pixels": int(selected_top_k),
        "selected_top_k_fraction": float(selected_top_k / num_pixels),
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
        "top_k_grid": candidates,
    }
    save_json(Path(log_dir) / "calibration_selection.json", selection)
    metrics = {
        **selection,
        **{f"calibration_{name}": value for name, value in calibration_ranking.items()},
        **{f"calibration_{name}": value for name, value in calibration_threshold.items()},
        "calibration_num_good": int((calibration_scores["label"] == 0).sum()),
        "calibration_num_anomaly": int((calibration_scores["label"] == 1).sum()),
        "calibration_num_source_groups": int(calibration_scores["source_group"].nunique()),
    }
    save_json(Path(log_dir) / "calibration_metrics.json", metrics)
    save_text_metrics(Path(log_dir) / "calibration_metrics.txt", metrics)
    return metrics


def load_calibration_selection(log_dir, config):
    selection_path = Path(log_dir) / "calibration_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Locked test requires completed calibration: {selection_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    required = {
        "image_size",
        "num_pixels",
        "selected_top_k_pixels",
        "selected_top_k_fraction",
        "threshold_quantile",
        "threshold",
    }
    missing = required - set(selection)
    if missing:
        raise ValueError(f"Calibration selection is missing keys: {sorted(missing)}")
    num_pixels = config["image_size"] ** 2
    if selection["image_size"] != config["image_size"] or selection["num_pixels"] != num_pixels:
        raise ValueError("Calibration selection does not match model input size")
    if not 1 <= selection["selected_top_k_pixels"] <= num_pixels:
        raise ValueError("Calibration selected_top_k_pixels is out of range")
    return selection


def evaluate_locked_test(
    model,
    test_dataset,
    config,
    log_dir,
    seed,
    device,
    bootstrap_iterations=2000,
):
    if bootstrap_iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    selection = load_calibration_selection(log_dir, config)
    selected_top_k = int(selection["selected_top_k_pixels"])
    score_column = f"score_top_{selected_top_k}"
    test_scores = collect_scores(
        model,
        test_dataset,
        [selected_top_k],
        device,
        config["batch_size"],
        seed,
        "locked test",
    ).rename(columns={score_column: "score"})
    test_scores.to_csv(Path(log_dir) / "test_scores.csv", index=False)

    test_ranking = multilevel_ranking_metrics(test_scores, "score")
    test_threshold = multilevel_threshold_metrics(
        test_scores,
        "score",
        selection["threshold"],
    )
    bootstrap = bootstrap_confidence_intervals(
        test_scores,
        selection["threshold"],
        iterations=int(bootstrap_iterations),
        seed=int(seed + 100_000),
    )
    save_json(Path(log_dir) / "test_bootstrap_ci.json", bootstrap)
    primary_ci = bootstrap["metrics"]["source_group_balanced_tile_roc_auc"]
    metrics = {
        **selection,
        **{f"test_{name}": value for name, value in test_ranking.items()},
        **{f"test_{name}": value for name, value in test_threshold.items()},
        "test_num_good": int((test_scores["label"] == 0).sum()),
        "test_num_anomaly": int((test_scores["label"] == 1).sum()),
        "test_num_source_groups": int(test_scores["source_group"].nunique()),
        "test_primary_roc_auc_ci_low": primary_ci["ci_low"],
        "test_primary_roc_auc_ci_high": primary_ci["ci_high"],
        "bootstrap_iterations": int(bootstrap_iterations),
    }
    save_json(Path(log_dir) / "test_metrics.json", metrics)
    save_text_metrics(Path(log_dir) / "test_metrics.txt", metrics)
    return metrics


def evaluate_fastflow(
    model,
    calibration_dataset,
    test_dataset,
    config,
    log_dir,
    seed,
    device,
    bootstrap_iterations=2000,
):
    calibration_metrics = calibrate_fastflow(
        model,
        calibration_dataset,
        config,
        log_dir,
        seed,
        device,
    )
    test_metrics = evaluate_locked_test(
        model,
        test_dataset,
        config,
        log_dir,
        seed,
        device,
        bootstrap_iterations=bootstrap_iterations,
    )
    return {**calibration_metrics, **test_metrics}


def calibration_has_artifacts(log_dir):
    artifact_names = {
        "calibration_top_k_sweep.csv",
        "calibration_top_k_scores.csv",
        "calibration_scores.csv",
        "calibration_selection.json",
        "calibration_metrics.json",
        "calibration_metrics.txt",
    }
    return any((Path(log_dir) / name).exists() for name in artifact_names)


def test_has_artifacts(log_dir):
    artifact_names = {
        "test_scores.csv",
        "test_metrics.json",
        "test_metrics.txt",
        "test_bootstrap_ci.json",
    }
    return any((Path(log_dir) / name).exists() for name in artifact_names)


def experiment_log_dir(experiments_root, config, try_number, seed):
    return (
        Path(experiments_root)
        / config["tag"]
        / f"try_{try_number}_seed_{seed}"
    )


def load_experiment_result(experiments_root, config, try_number, seed):
    log_dir = experiment_log_dir(experiments_root, config, try_number, seed)
    candidates = (
        ("test", log_dir / "test_metrics.json"),
        ("calibration", log_dir / "calibration_metrics.json"),
    )
    stage, metrics_path = next(
        ((stage, path) for stage, path in candidates if path.is_file()),
        (None, None),
    )
    if metrics_path is None:
        raise FileNotFoundError(f"No calibration/test metrics found in {log_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "config_name": config["name"],
        "seed": int(seed),
        "stage": stage,
        "log_dir": str(log_dir),
        **metrics,
    }


def run_experiment(
    config,
    seed,
    dataset_root,
    manifest_path,
    split_config_path,
    experiments_root,
    try_number,
    run_training=True,
    run_calibration=True,
    run_test=False,
    allow_overwrite=False,
    deterministic=True,
    verify_manifest_hashes=False,
    bootstrap_iterations=2000,
):
    if not any((run_training, run_calibration, run_test)):
        raise ValueError("At least one experiment stage must be enabled")
    seed_everything(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, split_config = load_approved_manifest(
        dataset_root,
        manifest_path,
        split_config_path,
        verify_hashes=verify_manifest_hashes,
    )
    log_dir = experiment_log_dir(experiments_root, config, try_number, seed)
    if (
        run_training
        and log_dir.exists()
        and any(log_dir.iterdir())
        and not allow_overwrite
    ):
        raise FileExistsError(f"Refusing to overwrite experiment artifacts: {log_dir}")
    if (
        not run_training
        and run_calibration
        and calibration_has_artifacts(log_dir)
        and not allow_overwrite
    ):
        raise FileExistsError(f"Refusing to overwrite calibration artifacts: {log_dir}")
    if run_test and test_has_artifacts(log_dir) and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite locked-test artifacts: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    train_transform = image_transform(config["image_size"], config.get("augmentation", "none"))
    eval_transform = image_transform(config["image_size"], "none")
    datasets = {
        split: ManifestDataset(
            dataset_root,
            manifest.loc[manifest["split"] == split],
            train_transform if split == "train" else eval_transform,
            include_metadata=split in {"calibration", "test"},
        )
        for split in ("train", "normal_val_loss", "calibration", "test")
    }
    run_config = {
        "config": config,
        "seed": seed,
        "try_number": try_number,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "split_version": split_config["version"],
        "manifest_sha256": file_sha256(manifest_path),
        "split_config_sha256": file_sha256(split_config_path),
        "pipeline_sha256": file_sha256(Path(__file__).resolve()),
        **git_provenance(Path(__file__).resolve().parents[1]),
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "split_class_counts": {
            f"{split}/{class_name}": int(len(rows))
            for (split, class_name), rows in manifest.groupby(["split", "class_name"])
        },
        "top_k_selection": (
            "calibration source-group-balanced tile ROC AUC; "
            "ties by object/tile ROC AUC, AP, then smaller k"
        ),
        "threshold_selection": (
            "calibration normal tile-score quantile only; the same tile threshold "
            "is applied after object/source max aggregation for operating metrics"
        ),
        "primary_metric": "source_group_balanced_tile_roc_auc",
        "bootstrap_iterations": int(bootstrap_iterations),
        "test_is_locked": True,
    }
    run_config_path = log_dir / "run_config.json"
    if run_config_path.exists() and not allow_overwrite:
        existing_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        consistency_keys = (
            "config",
            "seed",
            "try_number",
            "split_version",
            "manifest_sha256",
            "split_config_sha256",
        )
        if any(existing_run_config.get(key) != run_config.get(key) for key in consistency_keys):
            raise ValueError("Existing run_config.json does not match the requested run")
    else:
        save_json(run_config_path, run_config)
        if run_training:
            write_run_note(log_dir / "run_note.md", run_config)
    print(json.dumps({"log_dir": str(log_dir), **run_config}, ensure_ascii=False, indent=2))

    weights_path = log_dir / "best_model_weights.pth"
    if run_training:
        model = build_fastflow_model(config)
        model = train_fastflow(
            model,
            datasets["train"],
            datasets["normal_val_loss"],
            config,
            log_dir,
            seed,
            device,
        )
    else:
        if not weights_path.exists():
            raise FileNotFoundError(weights_path)
        model = build_fastflow_model(config).to(device)
        load_model_weights(model, weights_path, device)

    result = {
        "config_name": config["name"],
        "seed": seed,
        "log_dir": str(log_dir),
    }
    if run_calibration:
        calibration_metrics = calibrate_fastflow(
            model,
            datasets["calibration"],
            config,
            log_dir,
            seed,
            device,
        )
        result.update(calibration_metrics)
    if run_test:
        test_metrics = evaluate_locked_test(
            model,
            datasets["test"],
            config,
            log_dir,
            seed,
            device,
            bootstrap_iterations=bootstrap_iterations,
        )
        result.update(test_metrics)
    return result
