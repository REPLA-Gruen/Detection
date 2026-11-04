#!/usr/bin/env python3
"""Shared training utilities for D1 Wagon/Container classification."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ultralytics_runtime import suppress_auxiliary_amp_model_check

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}
EXPECTED_CLASSES = {"wagon", "container"}


def normalized_name(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def find_split_dir(root: Path, split: str) -> Path | None:
    for alias in SPLIT_ALIASES[split]:
        candidate = root / alias
        if candidate.is_dir():
            return candidate
    return None


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS
    )


def inspect_classification_dataset(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"D1 dataset folder does not exist: {root}")

    report: dict[str, Any] = {"dataset_root": str(root), "splits": {}, "classes": []}
    all_classes: set[str] = set()

    for split in ("train", "val", "test"):
        split_dir = find_split_dir(root, split)
        if split_dir is None:
            raise FileNotFoundError(f"Missing D1 split folder: {split}")
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if not class_dirs:
            raise RuntimeError(f"No class folders found in {split_dir}")

        counts: dict[str, int] = {}
        for class_dir in class_dirs:
            count = len(list_images(class_dir))
            if count == 0:
                raise RuntimeError(f"Class folder is empty: {class_dir}")
            counts[class_dir.name] = count
            all_classes.add(normalized_name(class_dir.name))
        report["splits"][split] = {
            "path": str(split_dir),
            "classes": counts,
            "images": sum(counts.values()),
        }

    if all_classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "D1 must contain exactly the classes Wagon and Container. "
            f"Found normalized class names: {sorted(all_classes)}"
        )

    report["classes"] = sorted(all_classes)
    return report


def environment_report() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        data["torch"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["cuda_version"] = getattr(torch.version, "cuda", None)
        data["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        data["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        data["gpus"] = [torch.cuda.get_device_name(i) for i in range(data["gpu_count"])]
    except Exception as exc:
        data["torch_error"] = str(exc)
    try:
        import ultralytics

        data["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        data["ultralytics_error"] = str(exc)
    return data


def augmentation_overrides(profile: str) -> dict[str, Any]:
    if profile == "default":
        return {}
    if profile == "none":
        return {
            "auto_augment": None,
            "erasing": 0.0,
            "scale": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "crop_fraction": 1.0,
        }
    if profile == "light":
        return {
            "auto_augment": None,
            "erasing": 0.0,
            "scale": 0.10,
            "fliplr": 0.50,
            "flipud": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.20,
            "hsv_v": 0.10,
            "degrees": 3.0,
            "translate": 0.05,
            "shear": 0.0,
            "perspective": 0.0,
            "crop_fraction": 1.0,
        }
    raise ValueError(f"Unknown augmentation profile: {profile}")


def add_training_arguments(parser: argparse.ArgumentParser, default_model: str, default_name: str) -> None:
    parser.add_argument("--data", type=Path, required=True, help="Created D1 dataset root.")
    parser.add_argument("--model", default=default_model, help="Ultralytics classification checkpoint.")
    parser.add_argument("--project", type=Path, required=True, help="Root folder for training runs.")
    parser.add_argument("--name", default=default_name, help="Run name under --project.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--augmentation", choices=("none", "light", "default"), default="none")
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N layers; 0 disables freezing.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")


def parse_cache(value: str) -> bool | str:
    return False if value == "false" else value


def train_classifier(args: argparse.Namespace, family: str) -> Path:
    dataset_report = inspect_classification_dataset(args.data)
    project = args.project.expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    intended_run_dir = project / args.name
    if intended_run_dir.exists() and not args.exist_ok:
        raise FileExistsError(
            f"Training run folder already exists: {intended_run_dir}. "
            "Use a new --name or add --exist-ok."
        )

    suppress_auxiliary_amp_model_check(True)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the active environment. "
            "Run: pip install ultralytics"
        ) from exc

    model = YOLO(args.model, task="classify")
    task = getattr(model, "task", None)
    if task not in (None, "classify"):
        raise RuntimeError(f"The supplied model is not a classification model: task={task!r}")

    train_args: dict[str, Any] = {
        "data": str(args.data.expanduser().resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(project),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "dropout": args.dropout,
        "cache": parse_cache(args.cache),
        "pretrained": not args.no_pretrained,
        "amp": not args.no_amp,
        "save": True,
        "save_period": args.save_period,
        "plots": True,
        "verbose": True,
        "val": True,
    }
    if args.freeze > 0:
        train_args["freeze"] = args.freeze
    train_args.update(augmentation_overrides(args.augmentation))

    requested = {
        "family": family,
        "model": args.model,
        "training_arguments": train_args,
        "dataset": dataset_report,
        "environment": environment_report(),
    }
    project.mkdir(parents=True, exist_ok=True)
    request_path = project / f"{args.name}_requested_training_config.json"
    request_path.write_text(json.dumps(requested, indent=2, ensure_ascii=False), encoding="utf-8")

    results = model.train(**train_args)
    save_dir_value = getattr(results, "save_dir", None)
    if save_dir_value is None and getattr(model, "trainer", None) is not None:
        save_dir_value = getattr(model.trainer, "save_dir", None)
    run_dir = Path(save_dir_value).resolve() if save_dir_value else intended_run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "requested_training_config.json").write_text(
        json.dumps(requested, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(requested["environment"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "dataset_statistics.json").write_text(
        json.dumps(dataset_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    print(f"Training completed: {run_dir}")
    print(f"Best checkpoint: {best}")
    print(f"Last checkpoint: {last}")
    return run_dir
