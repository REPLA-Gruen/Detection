#!/usr/bin/env python3
"""Shared training utilities for T-Core+Other T=400 multiclass detection."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}


def normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def names_to_dict(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in sorted(raw.items(), key=lambda item: int(item[0]))}
    raise RuntimeError("dataset.yaml field 'names' must be a list or mapping")


def find_yaml(root: Path) -> Path:
    for name in ("dataset.yaml", "dataset.yml", "data.yaml", "data.yml"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    candidates = sorted((*root.glob("*.yaml"), *root.glob("*.yml")))
    if not candidates:
        raise FileNotFoundError(f"No dataset YAML/YML found under: {root}")
    return candidates[0]


def resolve_detection_dataset(value: Path) -> tuple[Path, Path]:
    value = value.expanduser().resolve()
    if value.is_file():
        if value.suffix.casefold() not in {".yaml", ".yml"}:
            raise ValueError(f"Expected a dataset YAML/YML or folder, received: {value}")
        return value.parent, value
    if not value.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {value}")
    if (value / "detection").is_dir():
        value = value / "detection"
    return value, find_yaml(value)


def split_dir(root: Path, kind: str, split: str) -> Path:
    base = root / kind
    for alias in SPLIT_ALIASES[split]:
        candidate = base / alias
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing {kind}/{split} under {root}")


def list_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS)


def corresponding_label(image: Path, images_root: Path, labels_root: Path) -> Path:
    return labels_root / image.relative_to(images_root).with_suffix(".txt")


def validate_names(names: dict[int, str]) -> None:
    """Validate a dynamically generated T-Core+Other taxonomy.

    The original Phase-4 helper was tied to T=400 and required Plastiksack.
    The final package supports any T, so only non-empty contiguous IDs are
    mandatory. The builder itself records whether an Other class was needed.
    """
    if not names:
        raise RuntimeError("Dataset has no classes")
    if sorted(names) != list(range(len(names))):
        raise RuntimeError(f"Class IDs must be contiguous from zero. Found: {names}")
    normalized = [normalize_name(name) for name in names.values()]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError(f"Class names are not unique after normalization: {names}")


def parse_detection_label(path: Path, names: dict[int, str]) -> tuple[list[int], list[str]]:
    if not path.is_file():
        return [], [f"missing_label: {path}"]
    classes: list[int] = []
    errors: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        tokens = raw.strip().split()
        if not tokens:
            continue
        try:
            if len(tokens) != 5:
                raise ValueError("detection row must contain class + 4 bbox values")
            class_value = float(tokens[0])
            class_id = int(class_value)
            if class_value != class_id or class_id not in names:
                raise ValueError(f"invalid class ID {tokens[0]}")
            xc, yc, width, height = [float(value) for value in tokens[1:]]
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                raise ValueError("bbox center is outside [0,1]")
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                raise ValueError("bbox width/height must be in (0,1]")
            if xc - width / 2 < -1e-6 or yc - height / 2 < -1e-6:
                raise ValueError("bbox starts outside image")
            if xc + width / 2 > 1 + 1e-6 or yc + height / 2 > 1 + 1e-6:
                raise ValueError("bbox ends outside image")
            classes.append(class_id)
        except Exception as exc:
            errors.append(f"{path}:{line_number}: {exc}")
    return classes, errors


def inspect_detection_dataset(data: Path) -> dict[str, Any]:
    root, yaml_path = resolve_detection_dataset(data)
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    names = names_to_dict(raw.get("names"))
    validate_names(names)
    report: dict[str, Any] = {
        "dataset_root": str(root), "dataset_yaml": str(yaml_path), "names": names,
        "splits": {}, "errors": [],
    }
    total_class_instances = Counter()
    total_class_images = Counter()
    for split in ("train", "val", "test"):
        images_root = split_dir(root, "images", split)
        labels_root = split_dir(root, "labels", split)
        images = list_images(images_root)
        if not images:
            raise RuntimeError(f"No images found in {images_root}")
        positive = negative = instances = missing_labels = 0
        class_instances = Counter()
        class_images = Counter()
        split_errors: list[str] = []
        image_keys: set[str] = set()
        for image in images:
            relative = image.relative_to(images_root)
            image_keys.add(str(relative.with_suffix("")).casefold())
            label = corresponding_label(image, images_root, labels_root)
            classes, errors = parse_detection_label(label, names)
            split_errors.extend(errors)
            missing_labels += int(not label.is_file())
            if classes:
                positive += 1
                instances += len(classes)
                class_instances.update(classes)
                class_images.update(set(classes))
            else:
                negative += 1
        for label in labels_root.rglob("*.txt"):
            key = str(label.relative_to(labels_root).with_suffix("")).casefold()
            if key not in image_keys:
                split_errors.append(f"orphan_label: {label}")
        report["splits"][split] = {
            "images": len(images), "positive_images": positive, "negative_images": negative,
            "instances": instances, "missing_labels": missing_labels,
            "class_instances": {str(k): class_instances[k] for k in names},
            "class_positive_images": {str(k): class_images[k] for k in names},
            "errors": len(split_errors),
        }
        report["errors"].extend(split_errors)
        total_class_instances.update(class_instances)
        total_class_images.update(class_images)
    report["total_images"] = sum(item["images"] for item in report["splits"].values())
    report["positive_images"] = sum(item["positive_images"] for item in report["splits"].values())
    report["negative_images"] = sum(item["negative_images"] for item in report["splits"].values())
    report["instances"] = sum(item["instances"] for item in report["splits"].values())
    report["class_instances"] = {str(k): total_class_instances[k] for k in names}
    report["class_positive_images"] = {str(k): total_class_images[k] for k in names}
    if report["errors"]:
        preview = "\n".join(report["errors"][:20])
        raise RuntimeError(f"Dataset validation found {len(report['errors'])} errors. First errors:\n{preview}")
    return report


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "executable": sys.executable}
    try:
        import torch
        report.update({
            "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(report["gpu_count"])]
    except Exception as exc:
        report["torch_error"] = str(exc)
    try:
        import ultralytics
        report["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        report["ultralytics_error"] = str(exc)
    return report


def copy_or_link(source: Path, target: Path, mode: str = "hardlink") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if mode == "hardlink":
        try:
            target.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def prepare_training_view(
    data: Path, output: Path, *, negative_policy: str, negative_ratio: float,
    seed: int, copy_mode: str, overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    """Retain every positive train image and control only negative train images.

    Validation and test always preserve their natural prevalence. This separates the
    immutable benchmark from run-specific negative sampling.
    """
    root, yaml_path = resolve_detection_dataset(data)
    stats = inspect_detection_dataset(root)
    names = {int(k): v for k, v in stats["names"].items()}
    output = output.expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Training view already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        source_images = split_dir(root, "images", split)
        source_labels = split_dir(root, "labels", split)
        images = list_images(source_images)
        positives: list[Path] = []
        negatives: list[Path] = []
        class_instances_available = Counter()
        class_images_available = Counter()
        cache: dict[Path, list[int]] = {}
        for image in images:
            label = corresponding_label(image, source_images, source_labels)
            classes, errors = parse_detection_label(label, names)
            if errors:
                raise RuntimeError("\n".join(errors))
            cache[image] = classes
            if classes:
                positives.append(image)
                class_instances_available.update(classes)
                class_images_available.update(set(classes))
            else:
                negatives.append(image)
        if split == "train" and negative_policy == "balanced":
            requested_negative = int(round(len(positives) * negative_ratio))
            selected_negatives = rng.sample(negatives, min(requested_negative, len(negatives)))
            selected = positives + selected_negatives
            rng.shuffle(selected)
        else:
            selected = list(images)
            selected_negatives = negatives
            requested_negative = len(negatives)
        selected_positive = selected_negative = selected_instances = 0
        class_instances_selected = Counter()
        class_images_selected = Counter()
        for image in selected:
            relative = image.relative_to(source_images)
            source_label = corresponding_label(image, source_images, source_labels)
            target_image = output / "images" / split / relative
            target_label = output / "labels" / split / relative.with_suffix(".txt")
            copy_or_link(image, target_image, copy_mode)
            copy_or_link(source_label, target_label, copy_mode)
            classes = cache[image]
            selected_positive += int(bool(classes))
            selected_negative += int(not classes)
            selected_instances += len(classes)
            class_instances_selected.update(classes)
            class_images_selected.update(set(classes))
            manifest.append({
                "split": split, "source_image": str(image), "source_label": str(source_label),
                "output_image": str(target_image), "output_label": str(target_label),
                "positive": bool(classes), "instances": len(classes),
                "classes": ";".join(str(x) for x in sorted(set(classes))),
            })
        split_summary[split] = {
            "available_images": len(images), "available_positive_images": len(positives),
            "available_negative_images": len(negatives), "selected_images": len(selected),
            "selected_positive_images": selected_positive, "selected_negative_images": selected_negative,
            "selected_instances": selected_instances, "requested_negative_images": requested_negative,
            "achieved_negative_positive_ratio": selected_negative / selected_positive if selected_positive else None,
            "available_class_instances": {str(k): class_instances_available[k] for k in names},
            "selected_class_instances": {str(k): class_instances_selected[k] for k in names},
            "available_class_positive_images": {str(k): class_images_available[k] for k in names},
            "selected_class_positive_images": {str(k): class_images_selected[k] for k in names},
        }
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump({
        "path": output.as_posix(), "train": "images/train", "val": "images/val",
        "test": "images/test", "names": names, "nc": len(names),
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with (output / "training_selection_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0].keys()))
        writer.writeheader(); writer.writerows(manifest)
    summary = {
        "source_dataset_root": str(root), "source_dataset_yaml": str(yaml_path),
        "training_view_root": str(output), "training_view_yaml": str(dataset_yaml),
        "train_negative_policy": negative_policy, "requested_negative_ratio": negative_ratio,
        "seed": seed, "copy_mode": copy_mode, "master_dataset_statistics": stats,
        "splits": split_summary,
        "scientific_note": "All positive train images retained; only train negatives sampled. Val/test natural.",
    }
    (output / "training_view_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset_yaml, summary


def augmentation_overrides(profile: str) -> dict[str, Any]:
    if profile == "default": return {}
    if profile == "none":
        return {k: 0.0 for k in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste")}
    if profile == "light":
        return {"hsv_h": 0.015, "hsv_s": 0.20, "hsv_v": 0.10, "degrees": 3.0,
                "translate": 0.05, "scale": 0.10, "shear": 0.0, "perspective": 0.0,
                "flipud": 0.0, "fliplr": 0.50, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0}
    raise ValueError(f"Unknown augmentation profile: {profile}")


def add_training_arguments(parser: argparse.ArgumentParser, default_model: str, default_name: str) -> None:
    parser.add_argument("--data", type=Path, required=True, help="T-Core+Other root, detection root, or dataset.yaml")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default=default_name)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--augmentation", choices=("none", "light", "default"), default="light")
    parser.add_argument("--train-negative-policy", choices=("balanced", "natural"), default="balanced")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--training-view-copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--no-overwrite-run", action="store_true")


def parse_cache(value: str) -> bool | str:
    return False if value == "false" else value


def train_detector(args: argparse.Namespace, family: str) -> Path:
    if args.negative_ratio < 0: raise ValueError("--negative-ratio must be non-negative")
    master_stats = inspect_detection_dataset(args.data)
    project = args.project.expanduser().resolve(); project.mkdir(parents=True, exist_ok=True)
    run_name_path = Path(args.name)
    if not args.name.strip() or run_name_path.is_absolute() or len(run_name_path.parts) != 1 or args.name in {".", ".."}:
        raise ValueError("--name must be one safe folder name")
    run_dir = project / args.name
    if run_dir.exists():
        if args.no_overwrite_run: raise FileExistsError(f"Run folder already exists: {run_dir}")
        shutil.rmtree(run_dir) if run_dir.is_dir() and not run_dir.is_symlink() else run_dir.unlink()
    training_view = project / "_training_views" / args.name
    training_yaml, training_summary = prepare_training_view(
        args.data, training_view, negative_policy=args.train_negative_policy,
        negative_ratio=args.negative_ratio, seed=args.seed,
        copy_mode=args.training_view_copy_mode, overwrite=True,
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics") from exc
    model = YOLO(args.model, task="detect")
    train_args: dict[str, Any] = {
        "data": str(training_yaml), "epochs": args.epochs, "patience": args.patience,
        "imgsz": args.imgsz, "batch": args.batch, "device": args.device, "workers": args.workers,
        "project": str(project), "name": args.name, "exist_ok": True, "seed": args.seed,
        "deterministic": not args.non_deterministic, "optimizer": args.optimizer,
        "lr0": args.lr0, "lrf": args.lrf, "momentum": args.momentum,
        "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr, "cache": parse_cache(args.cache),
        "pretrained": not args.no_pretrained, "amp": not args.no_amp,
        "save": True, "save_period": args.save_period, "plots": True,
        "verbose": True, "val": True, "close_mosaic": 0, "max_det": args.max_det,
    }
    if args.freeze > 0: train_args["freeze"] = args.freeze
    train_args.update(augmentation_overrides(args.augmentation))
    requested = {
        "family": family, "model": args.model, "training_arguments": train_args,
        "master_dataset": master_stats, "training_view": training_summary,
        "environment": environment_report(),
        "controlled_settings": {
            "all_positive_train_images": True, "train_negative_ratio": args.negative_ratio,
            "validation_test_natural": True, "confidence_selected_on_validation_only": True,
        },
    }
    (project / f"{args.name}_requested_training_config.json").write_text(json.dumps(requested, indent=2, ensure_ascii=False), encoding="utf-8")
    results = model.train(**train_args)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and getattr(model, "trainer", None) is not None:
        save_dir = getattr(model.trainer, "save_dir", None)
    actual = Path(save_dir).resolve() if save_dir else run_dir.resolve(); actual.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("requested_training_config.json", requested), ("environment.json", requested["environment"]),
        ("master_dataset_statistics.json", master_stats), ("training_view_summary.json", training_summary),
    ):
        (actual / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Training completed: {actual}")
    print(f"Best checkpoint: {actual / 'weights' / 'best.pt'}")
    return actual
