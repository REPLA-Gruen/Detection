#!/usr/bin/env python3
"""Comprehensive evaluation for arbitrary T-Core+Other YOLO detectors.

One inference pass produces:
1. native Ultralytics IoU-based validation metrics;
2. same-class positive-overlap operational object metrics;
3. per-class image-presence metrics;
4. any-foreign-object image alarm metrics;
5. at-least-one-correct-detection image metrics;
6. a class-agnostic object confusion matrix with Background/no-object;
7. raw and row-normalized matrices, CSV files, match logs, and renderings.

Class names and class count are read dynamically from dataset.yaml. The code is
not tied to T=400 or to a fixed class taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from tcore_train_common import (
    corresponding_label,
    list_images,
    names_to_dict,
    normalize_name,
    resolve_detection_dataset,
    split_dir,
    validate_names,
)

PALETTE = [
    (255, 72, 72), (55, 190, 105), (65, 135, 255), (255, 190, 60),
    (180, 90, 255), (40, 205, 205), (255, 100, 190), (150, 210, 65),
    (245, 125, 35), (120, 150, 230), (225, 65, 135), (90, 205, 145),
]


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        array = np.asarray(value)
        if array.size != 1:
            return None
        result = float(array.reshape(-1)[0])
        return result if math.isfinite(result) else None
    except Exception:
        return None


def safe_array(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return [float(item) for item in np.asarray(value).reshape(-1)]
    except Exception:
        return []


def sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def get_font(size: int):
    for candidate in (
        r"C:\Windows\Fonts\arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_ground_truth(
    path: Path,
    image_size: tuple[int, int],
    names: dict[int, str],
) -> list[dict[str, Any]]:
    width, height = image_size
    if not path.is_file():
        raise FileNotFoundError(f"Missing label: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        tokens = raw.strip().split()
        if not tokens:
            continue
        if len(tokens) != 5:
            raise RuntimeError(
                f"Invalid detection row at {path}:{line_number}; "
                f"expected 5 tokens, found {len(tokens)}"
            )

        class_value = float(tokens[0])
        class_id = int(class_value)
        if class_value != class_id or class_id not in names:
            raise RuntimeError(f"Invalid class at {path}:{line_number}")

        xc, yc, box_width, box_height = [float(value) for value in tokens[1:]]
        if not (
            0.0 <= xc <= 1.0
            and 0.0 <= yc <= 1.0
            and 0.0 < box_width <= 1.0
            and 0.0 < box_height <= 1.0
        ):
            raise RuntimeError(f"Invalid normalized box at {path}:{line_number}")

        rows.append(
            {
                "class_id": class_id,
                "bbox": [
                    max(0.0, (xc - box_width / 2.0) * width),
                    max(0.0, (yc - box_height / 2.0) * height),
                    min(float(width), (xc + box_width / 2.0) * width),
                    min(float(height), (yc + box_height / 2.0) * height),
                ],
            }
        )
    return rows


def intersection_area(a: list[float], b: list[float]) -> float:
    return (
        max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    )


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = intersection_area(a, b)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def same_class_match(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, float, float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            if gt["class_id"] != prediction["class_id"]:
                continue
            intersection = intersection_area(gt["bbox"], prediction["bbox"])
            if intersection <= 0.0:
                continue
            candidates.append(
                (
                    box_iou(gt["bbox"], prediction["bbox"]),
                    intersection,
                    float(prediction["confidence"]),
                    gt_index,
                    pred_index,
                )
            )

    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []

    for iou, intersection, confidence, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(
            {
                "gt_index": gt_index,
                "prediction_index": pred_index,
                "class_id": ground_truth[gt_index]["class_id"],
                "prediction_confidence": confidence,
                "iou": iou,
                "intersection_area_px2": intersection,
            }
        )
    return matches, used_gt, used_pred


def class_agnostic_match(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    mode: str,
    match_iou: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, float, float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            intersection = intersection_area(gt["bbox"], prediction["bbox"])
            if intersection <= 0.0:
                continue
            iou = box_iou(gt["bbox"], prediction["bbox"])
            if mode == "iou" and iou < match_iou:
                continue
            candidates.append(
                (
                    iou,
                    intersection,
                    float(prediction["confidence"]),
                    gt_index,
                    pred_index,
                )
            )

    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []

    for iou, intersection, confidence, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)

        gt_class = int(ground_truth[gt_index]["class_id"])
        pred_class = int(predictions[pred_index]["class_id"])
        matches.append(
            {
                "gt_index": gt_index,
                "prediction_index": pred_index,
                "gt_class_id": gt_class,
                "prediction_class_id": pred_class,
                "prediction_confidence": confidence,
                "iou": iou,
                "intersection_area_px2": intersection,
                "class_correct": gt_class == pred_class,
            }
        )
    return matches, used_gt, used_pred


def binary_metrics(
    true_values: list[int],
    predicted_values: list[int],
) -> tuple[dict[str, Any], np.ndarray]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for true_value, predicted_value in zip(true_values, predicted_values):
        matrix[int(true_value), int(predicted_value)] += 1

    tn, fp = [int(value) for value in matrix[0]]
    fn, tp = [int(value) for value in matrix[1]]

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )

    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": safe_ratio(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "matthews_correlation_coefficient": safe_ratio(
            tp * tn - fp * fn,
            mcc_denominator,
        ),
        "negative_predictive_value": safe_ratio(tn, tn + fn),
        "false_positive_rate": safe_ratio(fp, fp + tn),
        "false_negative_rate": safe_ratio(fn, fn + tp),
    }, matrix


def extract_native(
    metrics: Any,
    names: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    box = getattr(metrics, "box", None)
    native = {
        "precision": safe_float(getattr(box, "mp", None)),
        "recall": safe_float(getattr(box, "mr", None)),
        "map50": safe_float(getattr(box, "map50", None)),
        "map75": safe_float(getattr(box, "map75", None)),
        "map50_95": safe_float(getattr(box, "map", None)),
    }

    precision = safe_array(getattr(box, "p", None))
    recall = safe_array(getattr(box, "r", None))
    f1 = safe_array(getattr(box, "f1", None))
    ap50 = safe_array(getattr(box, "ap50", None))
    ap = safe_array(getattr(box, "ap", None))
    maps = safe_array(getattr(box, "maps", None))
    class_indices = safe_array(getattr(box, "ap_class_index", None))
    class_to_metric_index = {
        int(class_id): index
        for index, class_id in enumerate(class_indices)
    }
    aligned = (
        not class_to_metric_index
        and max(len(precision), len(recall), len(ap50), len(ap), 0)
        == len(names)
    )

    rows: list[dict[str, Any]] = []
    for class_id, class_name in names.items():
        metric_index = class_to_metric_index.get(
            class_id,
            class_id if aligned else -1,
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": (
                    precision[metric_index]
                    if 0 <= metric_index < len(precision)
                    else None
                ),
                "recall": (
                    recall[metric_index]
                    if 0 <= metric_index < len(recall)
                    else None
                ),
                "f1": (
                    f1[metric_index]
                    if 0 <= metric_index < len(f1)
                    else None
                ),
                "ap50": (
                    ap50[metric_index]
                    if 0 <= metric_index < len(ap50)
                    else None
                ),
                "ap50_95": (
                    ap[metric_index]
                    if 0 <= metric_index < len(ap)
                    else maps[class_id] if class_id < len(maps) else None
                ),
            }
        )
    return native, rows


def runtime_yaml(
    root: Path,
    names: dict[int, str],
    output: Path,
) -> Path:
    payload = {
        "path": root.resolve().as_posix(),
        "train": split_dir(root, "images", "train").relative_to(root).as_posix(),
        "val": split_dir(root, "images", "val").relative_to(root).as_posix(),
        "test": split_dir(root, "images", "test").relative_to(root).as_posix(),
        "names": names,
        "nc": len(names),
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def plot_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output: Path,
    *,
    title: str,
    normalized: bool,
) -> None:
    values = matrix.astype(float if normalized else np.int64)
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values, dtype=float),
            where=row_sums != 0,
        )

    count = len(labels)
    figure_size = max(6.5, min(20.0, count * 1.2 + 3.0))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set(
        xticks=np.arange(count),
        yticks=np.arange(count),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="Ground-truth class",
        title=title,
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    maximum = float(values.max()) if values.size else 0.0
    threshold = maximum / 2.0 if maximum > 0.0 else 0.0

    for row_index in range(count):
        for column_index in range(count):
            value = values[row_index, column_index]
            label = f"{value:.3f}" if normalized else str(int(value))
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=max(7, 12 - count // 3),
            )

    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    labels: list[str],
    *,
    normalized: bool,
) -> None:
    values = matrix.astype(float if normalized else np.int64)
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values, dtype=float),
            where=row_sums != 0,
        )

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["GT \\ Predicted", *labels])
        for label, row in zip(labels, values):
            writer.writerow(
                [
                    label,
                    *(
                        [f"{float(value):.8f}" for value in row]
                        if normalized
                        else [int(value) for value in row]
                    ),
                ]
            )


def write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten(child, name))
    elif isinstance(value, list):
        rows.append((prefix, json.dumps(value, ensure_ascii=False)))
    else:
        rows.append((prefix, value))
    return rows


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: Any,
    color: tuple[int, int, int, int],
) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        (x, y, box[2] + 2 * pad, box[3] + 2 * pad),
        fill=color,
    )
    draw.text(
        (x + pad, y + pad),
        text,
        font=font,
        fill=(255, 255, 255, 255),
    )


def render_image(
    image_path: Path,
    output: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    names: dict[int, str],
    category: str,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    line_width = max(2, round(min(width, height) / 320))
    font = get_font(max(12, round(min(width, height) / 52)))
    title_font = get_font(max(14, round(min(width, height) / 40)))

    matched_gt = {match["gt_index"] for match in matches}
    matched_predictions = {
        match["prediction_index"] for match in matches
    }

    for index, gt in enumerate(ground_truth):
        color = (*PALETTE[gt["class_id"] % len(PALETTE)], 255)
        x1, y1, x2, y2 = gt["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width + 1)
        draw_label(
            draw,
            (int(x1), max(38, int(y1) - 25)),
            (
                f"GT {names[gt['class_id']]} "
                f"{'MATCH' if index in matched_gt else 'MISS'}"
            ),
            font,
            color,
        )

    for index, prediction in enumerate(predictions):
        base = PALETTE[prediction["class_id"] % len(PALETTE)]
        color = (
            (*base, 255)
            if index in matched_predictions
            else (220, 55, 55, 255)
        )
        x1, y1, x2, y2 = prediction["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width + 1)
        draw_label(
            draw,
            (int(x1), min(height - 25, int(y1) + 2)),
            (
                f"PRED {names[prediction['class_id']]} "
                f"{prediction['confidence']:.3f}"
            ),
            font,
            color,
        )

    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    draw.rectangle((0, 0, width, 36), fill=(20, 20, 20, 220))
    draw.text(
        (8, 5),
        (
            f"{category} | GT={len(ground_truth)} "
            f"PRED={len(predictions)} TP={true_positive} "
            f"FP={false_positive} FN={false_negative}"
        ),
        font=title_font,
        fill=(255, 255, 255, 255),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output = (
        output
        if output.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        else output.with_suffix(".jpg")
    )
    image.save(output, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive T-Core+Other YOLO evaluation."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--val-conf",
        type=float,
        default=0.001,
        help="Confidence floor for native Ultralytics validation.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--geometric-match-mode",
        choices=("positive_overlap", "iou"),
        default="positive_overlap",
    )
    parser.add_argument("--geometric-match-iou", type=float, default=0.50)
    parser.add_argument(
        "--background-label",
        default="Background / no object",
    )
    parser.add_argument(
        "--render",
        choices=("all", "errors", "none"),
        default="all",
    )
    parser.add_argument("--max-render", type=int, default=0)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--skip-native-val", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be in [0,1]")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in [0,1]")
    if not 0.0 <= args.geometric_match_iou <= 1.0:
        raise ValueError("--geometric-match-iou must be in [0,1]")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")

    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {output}. Use --overwrite to replace it."
            )
        shutil.rmtree(output)

    reports = output / "reports"
    reports.mkdir(parents=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements.txt."
        ) from exc

    dataset_root, dataset_yaml = resolve_detection_dataset(args.data)
    raw_yaml = yaml.safe_load(
        dataset_yaml.read_text(encoding="utf-8-sig")
    ) or {}
    names = names_to_dict(raw_yaml.get("names"))
    validate_names(names)

    images_root = split_dir(dataset_root, "images", args.split)
    labels_root = split_dir(dataset_root, "labels", args.split)
    images = list_images(images_root)
    if not images:
        raise RuntimeError(f"No images found under: {images_root}")

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path), task="detect")
    model_names = names_to_dict(model.names)
    if (
        {key: normalize_name(value) for key, value in model_names.items()}
        != {key: normalize_name(value) for key, value in names.items()}
    ):
        raise RuntimeError(
            f"Model names {model_names} do not match dataset names {names}"
        )

    absolute_yaml = runtime_yaml(
        dataset_root,
        names,
        reports / "dataset_absolute_runtime.yaml",
    )

    native_metrics: dict[str, Any] = {}
    native_rows: list[dict[str, Any]] = []
    if not args.skip_native_val:
        validation = model.val(
            data=str(absolute_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.val_conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            half=args.half,
            plots=True,
            project=str(output),
            name="ultralytics_val",
            exist_ok=True,
            verbose=True,
        )
        native_metrics, native_rows = extract_native(validation, names)

    number_of_classes = len(names)
    background_index = number_of_classes
    object_matrix_labels = [
        names[class_id] for class_id in range(number_of_classes)
    ] + [args.background_label]
    object_matrix = np.zeros(
        (number_of_classes + 1, number_of_classes + 1),
        dtype=np.int64,
    )

    totals = Counter()
    same_class_totals = {
        class_id: Counter() for class_id in names
    }
    per_class_true_presence = {
        class_id: [] for class_id in names
    }
    per_class_predicted_presence = {
        class_id: [] for class_id in names
    }

    any_alarm_true: list[int] = []
    any_alarm_predicted: list[int] = []
    correct_detection_true: list[int] = []
    correct_detection_predicted: list[int] = []

    per_image_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    same_class_match_rows: list[dict[str, Any]] = []
    geometric_match_rows: list[dict[str, Any]] = []

    rendered = 0
    total_wall_seconds = 0.0
    total_inference_ms = 0.0

    for batch_start in range(0, len(images), args.batch):
        batch_paths = images[batch_start : batch_start + args.batch]
        sync_cuda()
        start_time = time.perf_counter()
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=args.imgsz,
            batch=min(args.batch, len(batch_paths)),
            device=args.device,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            half=args.half,
            augment=args.augment,
            verbose=False,
            save=False,
        )
        sync_cuda()
        total_wall_seconds += time.perf_counter() - start_time

        if len(results) != len(batch_paths):
            raise RuntimeError("Prediction count does not match image count")

        for image_path, result in zip(batch_paths, results):
            relative = image_path.relative_to(images_root)
            label_path = corresponding_label(
                image_path,
                images_root,
                labels_root,
            )

            with Image.open(image_path) as source:
                image_size = source.size
            ground_truth = parse_ground_truth(
                label_path,
                image_size,
                names,
            )

            predictions: list[dict[str, Any]] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().float().cpu().numpy()
                classes = (
                    boxes.cls.detach().float().cpu().numpy().astype(int)
                )
                confidences = boxes.conf.detach().float().cpu().numpy()

                for prediction_index in range(len(xyxy)):
                    class_id = int(classes[prediction_index])
                    if class_id not in names:
                        raise RuntimeError(
                            f"Model predicted unknown class ID {class_id}"
                        )
                    prediction = {
                        "class_id": class_id,
                        "confidence": float(
                            confidences[prediction_index]
                        ),
                        "bbox": [
                            float(value)
                            for value in xyxy[prediction_index]
                        ],
                    }
                    predictions.append(prediction)
                    prediction_rows.append(
                        {
                            "image": str(image_path),
                            "relative_path": relative.as_posix(),
                            "prediction_index": prediction_index,
                            "class_id": class_id,
                            "class_name": names[class_id],
                            "confidence": prediction["confidence"],
                            "x1": prediction["bbox"][0],
                            "y1": prediction["bbox"][1],
                            "x2": prediction["bbox"][2],
                            "y2": prediction["bbox"][3],
                        }
                    )

            total_inference_ms += float(
                (getattr(result, "speed", {}) or {}).get(
                    "inference",
                    0.0,
                )
                or 0.0
            )

            same_matches, same_matched_gt, same_matched_predictions = (
                same_class_match(ground_truth, predictions)
            )
            geometric_matches, geometric_matched_gt, geometric_matched_pred = (
                class_agnostic_match(
                    ground_truth,
                    predictions,
                    mode=args.geometric_match_mode,
                    match_iou=args.geometric_match_iou,
                )
            )

            # General object confusion matrix.
            correct_geometric_classes = 0
            wrong_geometric_classes = 0
            for match in geometric_matches:
                gt_class = int(match["gt_class_id"])
                prediction_class = int(match["prediction_class_id"])
                object_matrix[gt_class, prediction_class] += 1
                if gt_class == prediction_class:
                    correct_geometric_classes += 1
                else:
                    wrong_geometric_classes += 1

                geometric_match_rows.append(
                    {
                        "image": str(image_path),
                        "relative_path": relative.as_posix(),
                        **match,
                        "gt_class_name": names[gt_class],
                        "prediction_class_name": names[prediction_class],
                    }
                )

            for gt_index, gt in enumerate(ground_truth):
                if gt_index not in geometric_matched_gt:
                    object_matrix[
                        int(gt["class_id"]),
                        background_index,
                    ] += 1

            for prediction_index, prediction in enumerate(predictions):
                if prediction_index not in geometric_matched_pred:
                    object_matrix[
                        background_index,
                        int(prediction["class_id"]),
                    ] += 1

            # Same-class operational metrics.
            for match in same_matches:
                same_class_match_rows.append(
                    {
                        "image": str(image_path),
                        "relative_path": relative.as_posix(),
                        **match,
                        "class_name": names[match["class_id"]],
                    }
                )

            gt_classes = {item["class_id"] for item in ground_truth}
            predicted_classes = {
                item["class_id"] for item in predictions
            }
            matched_classes = {
                match["class_id"] for match in same_matches
            }

            for class_id in names:
                gt_count = sum(
                    item["class_id"] == class_id
                    for item in ground_truth
                )
                prediction_count = sum(
                    item["class_id"] == class_id
                    for item in predictions
                )
                true_positive_count = sum(
                    match["class_id"] == class_id
                    for match in same_matches
                )
                same_class_totals[class_id].update(
                    {
                        "gt": gt_count,
                        "pred": prediction_count,
                        "tp": true_positive_count,
                        "fp": prediction_count - true_positive_count,
                        "fn": gt_count - true_positive_count,
                    }
                )
                per_class_true_presence[class_id].append(
                    int(class_id in gt_classes)
                )
                per_class_predicted_presence[class_id].append(
                    int(
                        class_id in matched_classes
                        if class_id in gt_classes
                        else class_id in predicted_classes
                    )
                )

            same_tp = len(same_matches)
            same_fp = len(predictions) - same_tp
            same_fn = len(ground_truth) - same_tp

            gt_present = bool(ground_truth)
            prediction_present = bool(predictions)
            correct_detection_present = bool(same_matches)

            any_alarm_true.append(int(gt_present))
            any_alarm_predicted.append(int(prediction_present))
            correct_detection_true.append(int(gt_present))
            correct_detection_predicted.append(
                int(
                    correct_detection_present
                    if gt_present
                    else prediction_present
                )
            )

            if not ground_truth and not predictions:
                category = "correct_negative"
            elif same_fp == 0 and same_fn == 0:
                category = "perfect_detection"
            elif same_fp > 0 and same_fn == 0:
                category = "false_positive_only"
            elif same_fn > 0 and same_fp == 0:
                category = "false_negative_only"
            else:
                category = "mixed_error"

            totals.update(
                {
                    "images": 1,
                    "positive_images": int(gt_present),
                    "negative_images": int(not gt_present),
                    "ground_truth_instances": len(ground_truth),
                    "predictions": len(predictions),
                    "same_class_tp": same_tp,
                    "same_class_fp": same_fp,
                    "same_class_fn": same_fn,
                    "geometric_matches": len(geometric_matches),
                    "correct_geometric_classes": correct_geometric_classes,
                    "wrong_geometric_classes": wrong_geometric_classes,
                    "unmatched_ground_truth": (
                        len(ground_truth) - len(geometric_matched_gt)
                    ),
                    "unmatched_predictions": (
                        len(predictions) - len(geometric_matched_pred)
                    ),
                    category: 1,
                }
            )

            per_image_rows.append(
                {
                    "image": str(image_path),
                    "relative_path": relative.as_posix(),
                    "ground_truth_instances": len(ground_truth),
                    "predictions": len(predictions),
                    "same_class_tp": same_tp,
                    "same_class_fp": same_fp,
                    "same_class_fn": same_fn,
                    "geometric_matches": len(geometric_matches),
                    "correct_geometric_classes": (
                        correct_geometric_classes
                    ),
                    "wrong_geometric_classes": wrong_geometric_classes,
                    "unmatched_ground_truth": (
                        len(ground_truth) - len(geometric_matched_gt)
                    ),
                    "unmatched_predictions": (
                        len(predictions) - len(geometric_matched_pred)
                    ),
                    "ground_truth_classes": ";".join(
                        names[class_id]
                        for class_id in sorted(gt_classes)
                    ),
                    "prediction_classes": ";".join(
                        names[class_id]
                        for class_id in sorted(predicted_classes)
                    ),
                    "matched_classes": ";".join(
                        names[class_id]
                        for class_id in sorted(matched_classes)
                    ),
                    "category": category,
                    "max_prediction_confidence": max(
                        (
                            prediction["confidence"]
                            for prediction in predictions
                        ),
                        default=0.0,
                    ),
                }
            )

            should_render = (
                args.render == "all"
                or (
                    args.render == "errors"
                    and (same_fp > 0 or same_fn > 0)
                )
            )
            if (
                should_render
                and (
                    args.max_render <= 0
                    or rendered < args.max_render
                )
            ):
                render_image(
                    image_path,
                    output / "rendered" / category / relative,
                    ground_truth,
                    predictions,
                    same_matches,
                    names,
                    category,
                )
                rendered += 1

        print(
            f"Processed {min(batch_start + args.batch, len(images))}/"
            f"{len(images)} images"
        )

    # Same-class operational metrics.
    micro_precision = safe_ratio(
        totals["same_class_tp"],
        totals["same_class_tp"] + totals["same_class_fp"],
    )
    micro_recall = safe_ratio(
        totals["same_class_tp"],
        totals["same_class_tp"] + totals["same_class_fn"],
    )
    micro_f1 = safe_ratio(
        2.0 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )

    per_class_rows: list[dict[str, Any]] = []
    for class_id, class_name in names.items():
        class_counts = same_class_totals[class_id]
        precision = safe_ratio(
            class_counts["tp"],
            class_counts["tp"] + class_counts["fp"],
        )
        recall = safe_ratio(
            class_counts["tp"],
            class_counts["tp"] + class_counts["fn"],
        )
        f1 = safe_ratio(
            2.0 * precision * recall,
            precision + recall,
        )
        presence_metrics, presence_matrix = binary_metrics(
            per_class_true_presence[class_id],
            per_class_predicted_presence[class_id],
        )
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "gt_instances": int(class_counts["gt"]),
            "predictions": int(class_counts["pred"]),
            "tp": int(class_counts["tp"]),
            "fp": int(class_counts["fp"]),
            "fn": int(class_counts["fn"]),
            "object_precision": precision,
            "object_recall": recall,
            "object_f1": f1,
            **{
                f"image_{key}": value
                for key, value in presence_metrics.items()
            },
        }
        per_class_rows.append(row)

        safe_name = "".join(
            char if char.isalnum() else "_"
            for char in class_name
        )
        plot_matrix(
            presence_matrix,
            [f"No {class_name}", class_name],
            reports / f"presence_{class_id:02d}_{safe_name}_counts.png",
            title=f"{class_name} image presence",
            normalized=False,
        )
        plot_matrix(
            presence_matrix,
            [f"No {class_name}", class_name],
            reports / f"presence_{class_id:02d}_{safe_name}_normalized.png",
            title=f"{class_name} image presence — row normalized",
            normalized=True,
        )

    macro_precision = safe_ratio(
        sum(row["object_precision"] for row in per_class_rows),
        len(per_class_rows),
    )
    macro_recall = safe_ratio(
        sum(row["object_recall"] for row in per_class_rows),
        len(per_class_rows),
    )
    macro_f1 = safe_ratio(
        sum(row["object_f1"] for row in per_class_rows),
        len(per_class_rows),
    )

    any_alarm_metrics, any_alarm_matrix = binary_metrics(
        any_alarm_true,
        any_alarm_predicted,
    )
    correct_detection_metrics, correct_detection_matrix = binary_metrics(
        correct_detection_true,
        correct_detection_predicted,
    )

    # General class-agnostic object metrics.
    geometric_precision = safe_ratio(
        totals["geometric_matches"],
        totals["predictions"],
    )
    geometric_recall = safe_ratio(
        totals["geometric_matches"],
        totals["ground_truth_instances"],
    )
    class_accuracy_given_localization = safe_ratio(
        totals["correct_geometric_classes"],
        totals["geometric_matches"],
    )
    class_aware_precision = safe_ratio(
        totals["correct_geometric_classes"],
        totals["predictions"],
    )
    class_aware_recall = safe_ratio(
        totals["correct_geometric_classes"],
        totals["ground_truth_instances"],
    )
    class_aware_f1 = safe_ratio(
        2.0 * class_aware_precision * class_aware_recall,
        class_aware_precision + class_aware_recall,
    )

    class_confusion_rows: list[dict[str, Any]] = []
    for class_id, class_name in names.items():
        gt_total = int(object_matrix[class_id, :].sum())
        correct = int(object_matrix[class_id, class_id])
        wrong_class = int(
            object_matrix[class_id, :number_of_classes].sum()
            - correct
        )
        missed = int(object_matrix[class_id, background_index])
        class_confusion_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_instances": gt_total,
                "correct_class_matches": correct,
                "wrong_class_matches": wrong_class,
                "missed_objects": missed,
                "correct_pct_of_gt": 100.0 * safe_ratio(correct, gt_total),
                "wrong_class_pct_of_gt": (
                    100.0 * safe_ratio(wrong_class, gt_total)
                ),
                "missed_pct_of_gt": 100.0 * safe_ratio(missed, gt_total),
            }
        )

    false_prediction_rows = []
    unmatched_total = int(object_matrix[background_index, :].sum())
    for class_id, class_name in names.items():
        count = int(object_matrix[background_index, class_id])
        false_prediction_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "unmatched_predictions": count,
                "pct_of_unmatched_predictions": (
                    100.0 * safe_ratio(count, unmatched_total)
                ),
            }
        )

    # Plots and matrix CSV files.
    plot_matrix(
        any_alarm_matrix,
        ["No FO", "FO present"],
        reports / "any_foreign_object_alarm_confusion_matrix.png",
        title="Any-foreign-object alarm",
        normalized=False,
    )
    plot_matrix(
        any_alarm_matrix,
        ["No FO", "FO present"],
        reports / "any_foreign_object_alarm_confusion_matrix_normalized.png",
        title="Any-foreign-object alarm — row normalized",
        normalized=True,
    )
    plot_matrix(
        correct_detection_matrix,
        ["No correctly detected FO", "Correctly detected FO"],
        reports / "correct_detection_presence_confusion_matrix.png",
        title="At least one correct-class positive-overlap detection",
        normalized=False,
    )
    plot_matrix(
        correct_detection_matrix,
        ["No correctly detected FO", "Correctly detected FO"],
        reports / "correct_detection_presence_confusion_matrix_normalized.png",
        title="Correct-detection presence — row normalized",
        normalized=True,
    )
    plot_matrix(
        object_matrix,
        object_matrix_labels,
        reports / "object_confusion_matrix_counts.png",
        title=(
            "Object-level multiclass confusion matrix "
            f"(conf={args.conf:g}, {args.geometric_match_mode})"
        ),
        normalized=False,
    )
    plot_matrix(
        object_matrix,
        object_matrix_labels,
        reports / "object_confusion_matrix_row_normalized.png",
        title=(
            "Object-level multiclass confusion matrix — row normalized "
            f"(conf={args.conf:g}, {args.geometric_match_mode})"
        ),
        normalized=True,
    )
    write_matrix_csv(
        reports / "object_confusion_matrix_counts.csv",
        object_matrix,
        object_matrix_labels,
        normalized=False,
    )
    write_matrix_csv(
        reports / "object_confusion_matrix_row_normalized.csv",
        object_matrix,
        object_matrix_labels,
        normalized=True,
    )

    metrics = {
        "configuration": {
            "model": str(model_path),
            "dataset": str(dataset_root),
            "dataset_yaml": str(dataset_yaml),
            "split": args.split,
            "classes": names,
            "confidence_threshold": args.conf,
            "native_validation_confidence_floor": args.val_conf,
            "nms_iou": args.nms_iou,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "geometric_match_mode": args.geometric_match_mode,
            "geometric_match_iou": (
                args.geometric_match_iou
                if args.geometric_match_mode == "iou"
                else None
            ),
        },
        "counts": {
            key: int(value) for key, value in totals.items()
        },
        "operating_point": {
            "matching_rule": (
                "same-class positive intersection (>0), greedy one-to-one "
                "matching ordered by descending IoU"
            ),
            "micro": {
                "tp": int(totals["same_class_tp"]),
                "fp": int(totals["same_class_fp"]),
                "fn": int(totals["same_class_fn"]),
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
        },
        "general_object_confusion": {
            "matching_rule": (
                "class-agnostic geometric matching, followed by class "
                "comparison"
            ),
            "geometric_localization_precision": geometric_precision,
            "geometric_localization_recall": geometric_recall,
            "class_accuracy_given_geometric_match": (
                class_accuracy_given_localization
            ),
            "class_aware_precision": class_aware_precision,
            "class_aware_recall": class_aware_recall,
            "class_aware_f1": class_aware_f1,
            "correct_class_matches": int(
                totals["correct_geometric_classes"]
            ),
            "wrong_class_matches": int(
                totals["wrong_geometric_classes"]
            ),
            "unmatched_ground_truth": int(
                totals["unmatched_ground_truth"]
            ),
            "unmatched_predictions": int(
                totals["unmatched_predictions"]
            ),
        },
        "any_foreign_object_alarm_presence": any_alarm_metrics,
        "at_least_one_correct_detection_presence": (
            correct_detection_metrics
        ),
        "native_ultralytics_validation": native_metrics,
        "runtime": {
            "wall_seconds": total_wall_seconds,
            "wall_ms_per_image": (
                1000.0
                * total_wall_seconds
                / max(totals["images"], 1)
            ),
            "ultralytics_inference_ms_per_image": (
                total_inference_ms / max(totals["images"], 1)
            ),
            "rendered_images": rendered,
        },
        "interpretation": [
            "Native metrics are conventional IoU-based YOLO metrics.",
            "Operational object metrics require correct class and any positive overlap.",
            "The general object confusion matrix matches geometry without using class, then records correct class, wrong class, miss, or unmatched prediction.",
            "Any-FO image alarm ignores localization and class correctness.",
        ],
    }

    (reports / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (reports / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(flatten(metrics))

    write_rows(
        reports / "native_per_class_metrics.csv",
        native_rows,
        [
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1",
            "ap50",
            "ap50_95",
        ],
    )
    write_rows(
        reports / "operational_per_class_metrics.csv",
        per_class_rows,
    )
    write_rows(
        reports / "object_class_confusion_summary.csv",
        class_confusion_rows,
    )
    write_rows(
        reports / "unmatched_prediction_distribution.csv",
        false_prediction_rows,
    )
    write_rows(
        reports / "per_image_results.csv",
        per_image_rows,
    )
    write_rows(
        reports / "predictions.csv",
        prediction_rows,
    )
    write_rows(
        reports / "same_class_object_matches.csv",
        same_class_match_rows,
    )
    write_rows(
        reports / "geometric_object_matches.csv",
        geometric_match_rows,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Comprehensive evaluation written to: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
