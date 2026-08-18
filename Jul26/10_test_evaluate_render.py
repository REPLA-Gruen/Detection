#!/usr/bin/env python3
"""Evaluate the final three-stage model on a YOLO split and render diagnostic results.

v1.3 rendering:
- GT and detections are drawn together.
- Prediction confidences use readable Det / Ver / Cls names.
- Rendered images are divided into:
    correct_negative
    perfect_detection
    false_positive_only
    false_negative_only
    mixed_error
- The category is based on same-class object matching, with class-uncertain
  verified FO proposals treated as unmatched predictions for diagnostic
  categorization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from phase4_inference import Pipeline, norm
from phase4_rendering import render_with_gt
from evaluate_TCoreOther_comprehensive import (
    parse_ground_truth,
    same_class_match,
    class_agnostic_match,
    binary_metrics,
    safe_ratio,
    plot_matrix,
    write_matrix_csv,
    write_rows,
)
from tcore_train_common import (
    resolve_detection_dataset,
    split_dir,
    list_images,
    corresponding_label,
    names_to_dict,
)

ROOT = Path(__file__).resolve().parent


def load(path):
    """Load a UTF-8/UTF-8-BOM JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fmt(value, threshold):
    """Resolve a configured path containing the optional {T} placeholder."""
    return Path(
        str(value).replace("{T}", str(threshold))
    ).expanduser().resolve()

CATEGORY_ORDER = (
    "correct_negative",
    "perfect_detection",
    "false_positive_only",
    "false_negative_only",
    "mixed_error",
)

def category_from_counts(
    gt_count,
    verified_count,
    same_tp,
    exact_prediction_count,
):
    """Replicate the earlier diagnostic categories for the three-stage model.

    Class-uncertain verified FO objects are counted as unmatched predictions
    for categorization, because they are valid image-level FO alarms but do not
    constitute a correct exact-class object detection.
    """
    uncertain_count = max(0, verified_count - exact_prediction_count)
    same_fp = max(0, exact_prediction_count - same_tp) + uncertain_count
    same_fn = max(0, gt_count - same_tp)

    if gt_count == 0 and verified_count == 0:
        return "correct_negative", same_fp, same_fn, uncertain_count
    if same_fp == 0 and same_fn == 0:
        return "perfect_detection", same_fp, same_fn, uncertain_count
    if same_fp > 0 and same_fn == 0:
        return "false_positive_only", same_fp, same_fn, uncertain_count
    if same_fn > 0 and same_fp == 0:
        return "false_negative_only", same_fp, same_fn, uncertain_count
    return "mixed_error", same_fp, same_fn, uncertain_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.json",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--operating-point", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render", choices=("none","errors","all"), default=None)
    args = parser.parse_args()

    config = load(args.config)
    threshold = int(config["threshold"])

    data = (
        args.data.expanduser().resolve()
        if args.data
        else fmt(config["dataset_output"], threshold) / "detection"
    )
    split = args.split or config["testing"]["split"]
    op_name = (
        args.operating_point
        or config["default_operating_point"]
    )
    settings = config["operating_points"][op_name]
    runs = fmt(config["runs_root"], threshold)
    models = (args.model_root.expanduser().resolve() if args.model_root else runs / "FINAL_MODELS")
    manifest = models / "final_model_manifest.json"

    if manifest.is_file():
        model_manifest = load(manifest)
        detector = model_manifest["models"]["stage_a"]
        verifier = model_manifest["models"]["stage_b"]
        exact = model_manifest["models"]["stage_c"]
    else:
        detector = models / "StageA_AnyFM_Detector" / "best.pt"
        verifier = models / "StageB_Binary_Verifier" / "best.pt"
        exact = models / "StageC_Exact_Classifier" / "best.pt"

    output = (
        args.output.expanduser().resolve()
        if args.output
        else runs / "TEST_RESULTS" / models.name / f"{split}_{op_name}"
    )
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output} exists; add --overwrite"
            )
        shutil.rmtree(output)

    render_mode = args.render or config["testing"].get("render", "all")
    reports = output / "reports"
    renders = output / "rendered"
    reports.mkdir(parents=True)
    for category in CATEGORY_ORDER:
        (renders / category).mkdir(parents=True, exist_ok=True)

    root, yaml_path = resolve_detection_dataset(data)
    raw = yaml.safe_load(
        yaml_path.read_text(encoding="utf-8-sig")
    )
    names = names_to_dict(raw["names"])
    images_root = split_dir(root, "images", split)
    labels_root = split_dir(root, "labels", split)
    images = list_images(images_root)

    pipeline = Pipeline(
        detector,
        verifier,
        exact,
        settings,
        config["device"],
    )

    number_classes = len(names)
    background = number_classes
    matrix = np.zeros(
        (number_classes + 1, number_classes + 1),
        dtype=np.int64,
    )
    totals = Counter()
    category_counts = Counter()
    per_class = {
        class_id: Counter()
        for class_id in names
    }
    y_true = []
    y_pred = []
    image_rows = []
    target_id_by_name = {
        norm(name): class_id
        for class_id, name in names.items()
    }

    for index, image_path in enumerate(images, 1):
        with Image.open(image_path) as source:
            size = source.size

        gt = parse_ground_truth(
            corresponding_label(
                image_path,
                images_root,
                labels_root,
            ),
            size,
            names,
        )
        image, result = pipeline.analyze(image_path)

        exact_predictions = []
        for detection in result["detections"]:
            if detection["class_name"] is None:
                continue
            mapped = target_id_by_name.get(
                norm(detection["class_name"])
            )
            if mapped is None:
                raise RuntimeError(
                    f"Exact classifier class "
                    f"{detection['class_name']!r} is not present "
                    f"in dataset taxonomy {names}"
                )
            exact_predictions.append(
                {
                    "class_id": int(mapped),
                    "bbox": detection["bbox"],
                    "confidence": (
                        detection["detector_confidence"]
                        * detection["verifier_fo_probability"]
                        * detection["exact_confidence"]
                    ),
                }
            )

        same_matches, _, _ = same_class_match(
            gt,
            exact_predictions,
        )
        geometric_matches, matched_gt, matched_pred = (
            class_agnostic_match(
                gt,
                exact_predictions,
                mode=config["testing"]["match_mode"],
                match_iou=float(
                    config["testing"]["match_iou"]
                ),
            )
        )

        for match in geometric_matches:
            matrix[
                int(match["gt_class_id"]),
                int(match["prediction_class_id"]),
            ] += 1
        for gt_index, ground_truth in enumerate(gt):
            if gt_index not in matched_gt:
                matrix[
                    int(ground_truth["class_id"]),
                    background,
                ] += 1
        for prediction_index, prediction in enumerate(
            exact_predictions
        ):
            if prediction_index not in matched_pred:
                matrix[
                    background,
                    int(prediction["class_id"]),
                ] += 1

        for class_id in names:
            gt_count = sum(
                int(item["class_id"]) == class_id
                for item in gt
            )
            prediction_count = sum(
                int(item["class_id"]) == class_id
                for item in exact_predictions
            )
            true_positive = sum(
                int(item["class_id"]) == class_id
                for item in same_matches
            )
            per_class[class_id].update(
                gt=gt_count,
                pred=prediction_count,
                tp=true_positive,
                fp=prediction_count - true_positive,
                fn=gt_count - true_positive,
            )

        gt_positive = bool(gt)
        predicted_positive = bool(result["detections"])
        y_true.append(int(gt_positive))
        y_pred.append(int(predicted_positive))

        correct_geometric_classes = sum(
            bool(item["class_correct"])
            for item in geometric_matches
        )

        totals.update(
            images=1,
            positive_images=int(gt_positive),
            negative_images=int(not gt_positive),
            ground_truth_instances=len(gt),
            verified_proposals=len(result["detections"]),
            exact_predictions=len(exact_predictions),
            same_class_tp=len(same_matches),
            same_class_fp=(
                len(exact_predictions) - len(same_matches)
            ),
            same_class_fn=len(gt) - len(same_matches),
            geometric_matches=len(geometric_matches),
            correct_geometric_classes=correct_geometric_classes,
            wrong_geometric_classes=(
                len(geometric_matches)
                - correct_geometric_classes
            ),
            unmatched_ground_truth=(
                len(gt) - len(matched_gt)
            ),
            unmatched_predictions=(
                len(exact_predictions) - len(matched_pred)
            ),
        )

        diagnostics = result.get("diagnostics", {})
        totals.update(
            invalid_detector_proposals_skipped=int(
                diagnostics.get(
                    "invalid_detector_proposals_skipped",
                    0,
                )
            ),
            invalid_verifier_crops_skipped=int(
                diagnostics.get(
                    "invalid_verifier_crops_skipped",
                    0,
                )
            ),
            invalid_exact_crops_skipped=int(
                diagnostics.get(
                    "invalid_exact_crops_skipped",
                    0,
                )
            ),
        )

        category, diagnostic_fp, diagnostic_fn, uncertain_count = (
            category_from_counts(
                len(gt),
                len(result["detections"]),
                len(same_matches),
                len(exact_predictions),
            )
        )
        category_counts[category] += 1

        render_path = None
        should_render = (
            render_mode == "all"
            or (render_mode == "errors" and category not in {"correct_negative", "perfect_detection"})
        )
        if should_render:
            render_path = (
                renders
                / category
                / f"{image_path.stem}__result.jpg"
            )
            rendered_image = render_with_gt(
                image,
                result,
                gt,
                exact_predictions,
                same_matches,
                names,
                category,
                diagnostic_fp,
                diagnostic_fn,
            )
            render_path.parent.mkdir(parents=True, exist_ok=True)
            rendered_image.save(render_path, quality=94)

        image_rows.append(
            {
                "image": str(image_path),
                "relative_path": image_path.relative_to(
                    images_root
                ).as_posix(),
                "render_category": category,
                "rendered_image": str(render_path) if render_path else "",
                "gt_instances": len(gt),
                "verified_fo_objects": len(
                    result["detections"]
                ),
                "exact_predictions": len(exact_predictions),
                "class_uncertain_verified_objects": uncertain_count,
                "same_class_tp": len(same_matches),
                "diagnostic_fp": diagnostic_fp,
                "diagnostic_fn": diagnostic_fn,
                "gt_image_positive": gt_positive,
                "predicted_image_positive": predicted_positive,
                "invalid_detector_proposals_skipped": int(
                    diagnostics.get(
                        "invalid_detector_proposals_skipped",
                        0,
                    )
                ),
                "invalid_verifier_crops_skipped": int(
                    diagnostics.get(
                        "invalid_verifier_crops_skipped",
                        0,
                    )
                ),
                "invalid_exact_crops_skipped": int(
                    diagnostics.get(
                        "invalid_exact_crops_skipped",
                        0,
                    )
                ),
            }
        )

        print(
            f"[{index}/{len(images)}] {image_path.name}: "
            f"GT={len(gt)} "
            f"FO={len(result['detections'])} "
            f"exact={len(exact_predictions)} "
            f"category={category}"
        )

    alarm, alarm_matrix = binary_metrics(
        y_true,
        y_pred,
    )

    micro_precision = safe_ratio(
        totals["same_class_tp"],
        totals["same_class_tp"]
        + totals["same_class_fp"],
    )
    micro_recall = safe_ratio(
        totals["same_class_tp"],
        totals["same_class_tp"]
        + totals["same_class_fn"],
    )
    micro_f1 = safe_ratio(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )

    class_rows = []
    for class_id, class_name in names.items():
        values = per_class[class_id]
        precision = safe_ratio(
            values["tp"],
            values["tp"] + values["fp"],
        )
        recall = safe_ratio(
            values["tp"],
            values["tp"] + values["fn"],
        )
        f1 = safe_ratio(
            2 * precision * recall,
            precision + recall,
        )
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_instances": values["gt"],
                "predictions": values["pred"],
                "tp": values["tp"],
                "fp": values["fp"],
                "fn": values["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = sum(
        row["f1"] for row in class_rows
    ) / len(class_rows)

    class_aware_precision = safe_ratio(
        totals["correct_geometric_classes"],
        totals["exact_predictions"],
    )
    class_aware_recall = safe_ratio(
        totals["correct_geometric_classes"],
        totals["ground_truth_instances"],
    )
    class_aware_f1 = safe_ratio(
        2 * class_aware_precision * class_aware_recall,
        class_aware_precision + class_aware_recall,
    )

    metrics = {
        "configuration": {
            "data": str(root),
            "split": split,
            "operating_point": op_name,
            "detector": str(detector),
            "verifier": str(verifier),
            "exact": str(exact),
            "rendering": {
                "gt_and_predictions_together": True,
                "categories": list(CATEGORY_ORDER),
                "render_mode": render_mode,
                "prediction_score_labels": {
                    "Det": "Stage-A detector confidence",
                    "Ver": (
                        "Stage-B ForeignMaterial verifier "
                        "probability"
                    ),
                    "Cls": (
                        "Stage-C exact-class confidence"
                    ),
                },
            },
        },
        "counts": dict(totals),
        "render_category_counts": {
            category: int(category_counts[category])
            for category in CATEGORY_ORDER
        },
        "same_class_operational": {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        },
        "general_object_confusion": {
            "geometric_localization_recall": safe_ratio(
                totals["geometric_matches"],
                totals["ground_truth_instances"],
            ),
            "class_accuracy_given_localization": safe_ratio(
                totals["correct_geometric_classes"],
                totals["geometric_matches"],
            ),
            "class_aware_precision": class_aware_precision,
            "class_aware_recall": class_aware_recall,
            "class_aware_f1": class_aware_f1,
        },
        "any_foreign_object_alarm": alarm,
    }

    (reports / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_rows(
        reports / "per_class_metrics.csv",
        class_rows,
    )
    write_rows(
        reports / "per_image_results.csv",
        image_rows,
    )

    category_rows = [
        {
            "category": category,
            "images": int(category_counts[category]),
            "percentage": safe_ratio(
                category_counts[category],
                totals["images"],
            )
            * 100.0,
        }
        for category in CATEGORY_ORDER
    ]
    write_rows(
        reports / "render_category_counts.csv",
        category_rows,
    )

    labels = [
        names[index]
        for index in range(number_classes)
    ] + ["Background / no object"]

    write_matrix_csv(
        reports / "object_confusion_matrix_counts.csv",
        matrix,
        labels,
        normalized=False,
    )
    write_matrix_csv(
        reports / "object_confusion_matrix_normalized.csv",
        matrix,
        labels,
        normalized=True,
    )
    plot_matrix(
        matrix,
        labels,
        reports / "object_confusion_matrix_counts.png",
        title="Three-stage object confusion matrix",
        normalized=False,
    )
    plot_matrix(
        matrix,
        labels,
        reports / "object_confusion_matrix_normalized.png",
        title="Three-stage object confusion matrix — normalized",
        normalized=True,
    )

    alarm_labels = ["No FO", "FO"]
    write_matrix_csv(
        reports / "any_fo_confusion_matrix_counts.csv",
        alarm_matrix,
        alarm_labels,
        normalized=False,
    )
    write_matrix_csv(
        reports / "any_fo_confusion_matrix_normalized.csv",
        alarm_matrix,
        alarm_labels,
        normalized=True,
    )
    plot_matrix(
        alarm_matrix,
        alarm_labels,
        reports / "any_fo_confusion_matrix_counts.png",
        title="Any-FO image confusion matrix",
        normalized=False,
    )
    plot_matrix(
        alarm_matrix,
        alarm_labels,
        reports / "any_fo_confusion_matrix_normalized.png",
        title="Any-FO image confusion matrix — normalized",
        normalized=True,
    )

    if config["testing"].get(
        "run_stage_a_native_metrics",
        True,
    ):
        try:
            import os

            native_root = output / "stage_a_native_dataset"
            native_images = native_root / "images" / split
            native_labels = native_root / "labels" / split
            native_images.mkdir(
                parents=True,
                exist_ok=True,
            )
            native_labels.mkdir(
                parents=True,
                exist_ok=True,
            )

            for source_image in images:
                relative = source_image.relative_to(
                    images_root
                )
                target_image = native_images / relative
                target_image.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                try:
                    os.link(
                        source_image,
                        target_image,
                    )
                except OSError:
                    shutil.copy2(
                        source_image,
                        target_image,
                    )

                source_label = corresponding_label(
                    source_image,
                    images_root,
                    labels_root,
                )
                target_label = (
                    native_labels / relative
                ).with_suffix(".txt")
                target_label.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                lines = []
                for raw_line in source_label.read_text(
                    encoding="utf-8-sig"
                ).splitlines():
                    tokens = raw_line.strip().split()
                    if tokens:
                        lines.append(
                            "0 " + " ".join(tokens[1:])
                        )
                target_label.write_text(
                    (
                        "\n".join(lines) + "\n"
                        if lines
                        else ""
                    ),
                    encoding="utf-8",
                )

            native_yaml = native_root / "dataset.yaml"
            native_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(native_root),
                        split: f"images/{split}",
                        "names": {0: "ForeignMaterial"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            native = pipeline.detector.val(
                data=str(native_yaml),
                split=split,
                imgsz=int(
                    settings["detector_imgsz"]
                ),
                batch=int(
                    config["testing"]["batch"]
                ),
                device=config["device"],
                conf=float(
                    settings[
                        "detector_inference_floor"
                    ]
                ),
                iou=float(settings["nms_iou"]),
                verbose=False,
                plots=True,
                project=str(output),
                name="stage_a_native",
                exist_ok=True,
            )

            box = native.box
            native_metrics = {
                "precision": float(box.mp),
                "recall": float(box.mr),
                "map50": float(box.map50),
                "map50_95": float(box.map),
            }
            (
                reports
                / "stage_a_native_metrics.json"
            ).write_text(
                json.dumps(
                    native_metrics,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            (
                reports
                / "stage_a_native_metrics_error.txt"
            ).write_text(
                str(exc),
                encoding="utf-8",
            )

    category_summary = "\n".join(
        (
            f"{category}: "
            f"{category_counts[category]}"
        )
        for category in CATEGORY_ORDER
    )

    (output / "SUMMARY.txt").write_text(
        (
            f"Images: {totals['images']}\n"
            f"Any-FO TN/FP/FN/TP: "
            f"{alarm['true_negative']}/"
            f"{alarm['false_positive']}/"
            f"{alarm['false_negative']}/"
            f"{alarm['true_positive']}\n"
            f"Specificity: {alarm['specificity']:.6f}\n"
            f"Any-FO recall: {alarm['recall']:.6f}\n"
            f"Balanced accuracy: "
            f"{alarm['balanced_accuracy']:.6f}\n"
            f"Same-class micro F1: {micro_f1:.6f}\n"
            f"Same-class macro F1: {macro_f1:.6f}\n"
            f"Class-aware F1: {class_aware_f1:.6f}\n"
            f"Invalid detector proposals skipped: "
            f"{totals['invalid_detector_proposals_skipped']}\n"
            f"Invalid verifier crops skipped: "
            f"{totals['invalid_verifier_crops_skipped']}\n"
            f"Invalid exact crops skipped: "
            f"{totals['invalid_exact_crops_skipped']}\n"
            "\nRendering categories:\n"
            f"{category_summary}\n"
            f"\nRendered images: {renders}\n"
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
