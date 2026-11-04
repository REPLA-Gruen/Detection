#!/usr/bin/env python3
"""Independent test evaluation and rendering for a trained D1 classifier."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def safe_slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_.-")
    return out or "class"


def find_split_dir(root: Path, split: str) -> Path:
    for alias in SPLIT_ALIASES[split]:
        candidate = root / alias
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Split folder {split!r} not found under {root}")


def list_split_images(root: Path, split: str) -> list[tuple[Path, str, Path]]:
    split_dir = find_split_dir(root, split)
    rows: list[tuple[Path, str, Path]] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"Class folder contains no images: {class_dir}")
        for image in images:
            rows.append((image, class_dir.name, image.relative_to(split_dir)))
    if not rows:
        raise RuntimeError(f"No images found in {split_dir}")
    return rows


def names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    raise RuntimeError(f"Unsupported model.names value: {type(names).__name__}")


def synchronize_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def get_font(size: int):
    from PIL import ImageFont

    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_prediction(
    image_path: Path,
    output_path: Path,
    true_name: str,
    pred_name: str,
    confidence: float,
    top_two: list[tuple[str, float]],
    model_label: str,
    correct: bool,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(image_path) as src:
        image = src.convert("RGB")
    width, height = image.size
    banner_h = max(118, round(height * 0.14))
    canvas = Image.new("RGB", (width, height + banner_h), (245, 245, 245))
    canvas.paste(image, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(max(16, round(min(width, height) / 36)))
    body_font = get_font(max(13, round(min(width, height) / 48)))
    small_font = get_font(max(11, round(min(width, height) / 58)))

    status = "CORRECT" if correct else "INCORRECT"
    status_fill = (35, 120, 55) if correct else (180, 40, 40)
    draw.rectangle((0, 0, width, banner_h), fill=(245, 245, 245))
    draw.rectangle((0, 0, width, max(30, banner_h // 4)), fill=status_fill)
    draw.text((10, 4), f"{status} | {model_label}", fill=(255, 255, 255), font=title_font)
    draw.text((10, banner_h // 4 + 6), f"Ground truth: {true_name}", fill=(20, 20, 20), font=body_font)
    draw.text((10, banner_h // 4 + 34), f"Prediction: {pred_name} | confidence={confidence:.4f}", fill=(20, 20, 20), font=body_font)
    top_text = "Top probabilities: " + " | ".join(f"{name}={prob:.4f}" for name, prob in top_two)
    draw.text((10, banner_h // 4 + 62), top_text, fill=(45, 45, 45), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.casefold() in {".jpg", ".jpeg"}:
        canvas.save(output_path, quality=94, subsampling=0)
    else:
        canvas.save(output_path.with_suffix(".png"), optimize=True)


def plot_confusion(cm: np.ndarray, labels: list[str], output: Path, normalized: bool) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    if normalized:
        values = cm.astype(float)
        denominator = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator != 0)
    else:
        values = cm.astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display = ConfusionMatrixDisplay(values, display_labels=labels)
    display.plot(ax=ax, values_format=".3f" if normalized else "d")
    ax.set_title("Normalized confusion matrix" if normalized else "Confusion matrix")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true_binary: np.ndarray, scores: np.ndarray, positive_name: str, output: Path) -> float:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(y_true_binary, scores))
    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, label=f"AUROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="No-skill")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"ROC curve: positive class = {positive_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return auc


def flatten_metrics(metrics: dict[str, Any]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    for item_key, item_value in subvalue.items():
                        rows.append((f"{key}.{subkey}.{item_key}", item_value))
                else:
                    rows.append((f"{key}.{subkey}", subvalue))
        elif not isinstance(value, (list, tuple)):
            rows.append((key, value))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and render a trained YOLO D1 classifier.")
    parser.add_argument("--model", type=Path, required=True, help="Trained classification checkpoint, normally best.pt.")
    parser.add_argument("--data", "--dataset-root", dest="data", type=Path, required=True, help="Created D1 root.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8, help="Reserved for compatibility and reporting.")
    parser.add_argument("--positive-class", default="container")
    parser.add_argument("--render", choices=("all", "errors", "none"), default="all")
    parser.add_argument("--max-render", type=int, default=0, help="0 renders all selected images.")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Use test-time augmentation during prediction.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            print(f"ERROR: Output exists: {output}; use --overwrite", file=sys.stderr)
            return 1
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_recall_fscore_support,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        print("ERROR: Install ultralytics, scikit-learn, matplotlib, Pillow, and NumPy.", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    data_root = args.data.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"ERROR: Model checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    samples = list_split_images(data_root, args.split)
    run_name = model_path.parents[1].name if model_path.parent.name.casefold() == "weights" and len(model_path.parents) > 1 else model_path.stem
    model_label = f"{run_name}/{model_path.name}"
    model = YOLO(str(model_path), task="classify")
    if getattr(model, "task", "classify") != "classify":
        print(f"ERROR: Model task is {getattr(model, 'task', None)!r}, not 'classify'.", file=sys.stderr)
        return 1

    class_names = names_to_dict(model.names)
    normalized_model_names = {normalize_name(name): class_id for class_id, name in class_names.items()}
    folder_names = sorted({true_name for _, true_name, _ in samples})
    unresolved = [name for name in folder_names if normalize_name(name) not in normalized_model_names]
    if unresolved:
        print(
            "ERROR: Dataset class folders do not match model class names. "
            f"Unresolved folders: {unresolved}; model names: {class_names}",
            file=sys.stderr,
        )
        return 1

    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[np.ndarray] = []
    prediction_rows: list[dict[str, Any]] = []
    total_wall = 0.0
    total_ultralytics_inference_ms = 0.0

    print(f"Model: {model_path}")
    print(f"Dataset: {data_root}")
    print(f"Split: {args.split} | images: {len(samples)}")
    print(f"Model classes: {class_names}")

    for start in range(0, len(samples), args.batch):
        batch_samples = samples[start:start + args.batch]
        batch_paths = [str(item[0]) for item in batch_samples]
        synchronize_cuda()
        tick = time.perf_counter()
        results = model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            batch=min(args.batch, len(batch_paths)),
            device=args.device,
            half=args.half,
            augment=args.augment,
            verbose=False,
            save=False,
        )
        synchronize_cuda()
        total_wall += time.perf_counter() - tick
        if len(results) != len(batch_samples):
            raise RuntimeError(f"Expected {len(batch_samples)} predictions, received {len(results)}")

        for (image_path, true_name, relative_path), result in zip(batch_samples, results):
            if result.probs is None:
                raise RuntimeError(f"Classification probabilities missing for {image_path}")
            prob_data = result.probs.data
            if hasattr(prob_data, "detach"):
                prob_array = prob_data.detach().float().cpu().numpy()
            else:
                prob_array = np.asarray(prob_data, dtype=float)
            prob_array = np.asarray(prob_array, dtype=float).reshape(-1)
            if prob_array.size != len(class_names):
                raise RuntimeError(
                    f"Probability count {prob_array.size} does not match model classes {len(class_names)}"
                )

            true_id = normalized_model_names[normalize_name(true_name)]
            pred_id = int(np.argmax(prob_array))
            pred_name = class_names[pred_id]
            confidence = float(prob_array[pred_id])
            top_ids = np.argsort(prob_array)[::-1][: min(2, len(prob_array))]
            top_two = [(class_names[int(i)], float(prob_array[int(i)])) for i in top_ids]
            correct = pred_id == true_id

            speed = getattr(result, "speed", {}) or {}
            inference_ms = float(speed.get("inference", 0.0) or 0.0)
            total_ultralytics_inference_ms += inference_ms

            row: dict[str, Any] = {
                "image": str(image_path),
                "relative_path": str(relative_path),
                "true_id": true_id,
                "true_class": true_name,
                "predicted_id": pred_id,
                "predicted_class": pred_name,
                "confidence": confidence,
                "correct": correct,
                "inference_ms": inference_ms,
                "top1_class": top_two[0][0],
                "top1_probability": top_two[0][1],
                "top2_class": top_two[1][0] if len(top_two) > 1 else "",
                "top2_probability": top_two[1][1] if len(top_two) > 1 else "",
            }
            for class_id, class_name in class_names.items():
                row[f"prob_{safe_slug(class_name)}"] = float(prob_array[class_id])
            prediction_rows.append(row)
            y_true.append(true_id)
            y_pred.append(pred_id)
            probabilities.append(prob_array)

            should_render = args.render == "all" or (args.render == "errors" and not correct)
            if should_render:
                if args.max_render <= 0 or sum(1 for r in prediction_rows if r.get("rendered")) < args.max_render:
                    if correct:
                        render_rel = Path("correct") / safe_slug(true_name) / relative_path.name
                    else:
                        render_rel = Path("incorrect") / f"{safe_slug(true_name)}_as_{safe_slug(pred_name)}" / relative_path.name
                    rendered_path = output / "rendered" / render_rel
                    render_prediction(
                        image_path,
                        rendered_path,
                        true_name,
                        pred_name,
                        confidence,
                        top_two,
                        model_label,
                        correct,
                    )
                    row["rendered"] = True
                    row["rendered_path"] = str(rendered_path)
                else:
                    row["rendered"] = False
                    row["rendered_path"] = ""
            else:
                row["rendered"] = False
                row["rendered_path"] = ""

        print(f"Processed {min(start + args.batch, len(samples))}/{len(samples)}")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    probability_matrix = np.vstack(probabilities)
    ordered_ids = sorted(class_names)
    ordered_labels = [class_names[i] for i in ordered_ids]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=ordered_ids)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=ordered_ids, zero_division=0
    )
    per_class = {
        class_names[class_id]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_id in enumerate(ordered_ids)
    }

    metrics: dict[str, Any] = {
        "model": str(model_path),
        "model_name": run_name,
        "checkpoint_name": model_path.name,
        "model_task": getattr(model, "task", None),
        "dataset": str(data_root),
        "split": args.split,
        "images": len(samples),
        "classes": class_names,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "correct": int(np.sum(y_true_arr == y_pred_arr)),
        "incorrect": int(np.sum(y_true_arr != y_pred_arr)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "wall_seconds": total_wall,
        "wall_ms_per_image": 1000.0 * total_wall / max(len(samples), 1),
        "ultralytics_inference_ms_per_image": total_ultralytics_inference_ms / max(len(samples), 1),
        "checkpoint_size_mb": model_path.stat().st_size / (1024.0 * 1024.0),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
        "test_time_augmentation": args.augment,
    }

    try:
        metrics["parameters"] = int(sum(parameter.numel() for parameter in model.model.parameters()))
        metrics["trainable_parameters"] = int(sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad))
    except Exception:
        metrics["parameters"] = None
        metrics["trainable_parameters"] = None

    positive_normalized = normalize_name(args.positive_class)
    if len(ordered_ids) == 2 and positive_normalized in normalized_model_names:
        positive_id = normalized_model_names[positive_normalized]
        y_binary = (y_true_arr == positive_id).astype(int)
        if len(np.unique(y_binary)) == 2:
            roc_auc = plot_roc(
                y_binary,
                probability_matrix[:, positive_id],
                class_names[positive_id],
                reports / "roc_curve.png",
            )
            metrics["positive_class"] = class_names[positive_id]
            metrics["roc_auc"] = roc_auc

    plot_confusion(cm, ordered_labels, reports / "confusion_matrix.png", normalized=False)
    plot_confusion(cm, ordered_labels, reports / "normalized_confusion_matrix.png", normalized=True)

    report_text = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=ordered_ids,
        target_names=ordered_labels,
        digits=6,
        zero_division=0,
    )
    (reports / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    with (reports / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key, value in flatten_metrics(metrics):
            writer.writerow([key, value])

    probability_fields = [f"prob_{safe_slug(class_names[class_id])}" for class_id in ordered_ids]
    fields = [
        "image", "relative_path", "true_id", "true_class", "predicted_id", "predicted_class",
        "confidence", "correct", "inference_ms", "top1_class", "top1_probability",
        "top2_class", "top2_probability", *probability_fields, "rendered", "rendered_path",
    ]
    with (reports / "per_image_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (reports / "misclassified_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in prediction_rows if not row["correct"])

    print(report_text)
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    print(f"MCC: {metrics['matthews_correlation_coefficient']:.6f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC ({metrics['positive_class']}): {metrics['roc_auc']:.6f}")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
