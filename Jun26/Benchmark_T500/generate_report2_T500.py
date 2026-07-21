#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Report 2 draft for YOLOv8m vs YOLO11m on final T500 benchmark.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval")


def main():
    overall_path = ROOT / "comparison_overall.csv"
    pc_path = ROOT / "comparison_per_class_map50_95.csv"

    if not overall_path.exists():
        raise FileNotFoundError(f"Missing: {overall_path}")
    if not pc_path.exists():
        raise FileNotFoundError(f"Missing: {pc_path}")

    overall = pd.read_csv(overall_path)
    pc = pd.read_csv(pc_path)

    name_map = {
        "yolov8m_T500_test": "YOLOv8m",
        "yolo11m_T500_test": "YOLO11m",
    }
    overall["Model"] = overall["model"].replace(name_map)
    pc["Model"] = pc["model"].replace(name_map)

    overall_table = overall[[
        "Model",
        "precision_B",
        "recall_B",
        "mAP50_B",
        "mAP50_95_B",
        "speed_inference_ms",
        "validation_wall_time_sec",
        "n_params",
    ]].copy()

    overall_table.columns = [
        "Model",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "Inference Speed (ms/image)",
        "Validation Wall Time (s)",
        "Parameters",
    ]

    pc_pivot = pc.pivot(index="class_name", columns="Model", values="map50_95").reset_index()
    pc_pivot.columns.name = None

    lines = []
    lines.append("# Report 2 — YOLOv8m vs YOLO11m Benchmark on the Final T500 Dataset")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "This report evaluates YOLOv8m and YOLO11m under identical experimental conditions on the finalized "
        "T500 learnability-aware benchmark dataset. The objective is to determine whether YOLO11m provides "
        "measurable improvements over the YOLOv8m baseline when the dataset has been stabilized through "
        "threshold-based grouping and stratified splitting."
    )
    lines.append("")
    lines.append("## 2. Benchmark Dataset")
    lines.append("")
    lines.append("- Dataset version: Jun-26 grouped T500 split v2")
    lines.append("- Classes: 6")
    lines.append("- Split: 80/10/10")
    lines.append("- Test split used for final evaluation")
    lines.append("- Class set: BAW-Sack, Plastic, Paper_Cardboard, Wagon, Container, Other")
    lines.append("")
    lines.append("## 3. Experimental Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---:|")
    lines.append("| Image size | 960 |")
    lines.append("| Epochs | 100 |")
    lines.append("| Batch size | 16 |")
    lines.append("| Seed | 42 |")
    lines.append("| Confidence threshold | 0.25 |")
    lines.append("| IoU threshold | 0.60 |")
    lines.append("| Device | CUDA GPU |")
    lines.append("")
    lines.append("## 4. Overall Quantitative Results")
    lines.append("")
    lines.append(overall_table.to_markdown(index=False))
    lines.append("")
    lines.append("## 5. Per-Class mAP50-95")
    lines.append("")
    lines.append(pc_pivot.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. Required Figures")
    lines.append("")
    lines.append("Insert the following evaluation figures:")
    lines.append("")
    lines.append("| Figure | File |")
    lines.append("|---|---|")
    lines.append("| YOLOv8m confusion matrix | `yolov8m_T500_test/confusion_matrix.png` |")
    lines.append("| YOLO11m confusion matrix | `yolo11m_T500_test/confusion_matrix.png` |")
    lines.append("| YOLOv8m normalized confusion matrix | `yolov8m_T500_test/confusion_matrix_normalized.png` |")
    lines.append("| YOLO11m normalized confusion matrix | `yolo11m_T500_test/confusion_matrix_normalized.png` |")
    lines.append("| YOLOv8m PR curve | `yolov8m_T500_test/PR_curve.png` |")
    lines.append("| YOLO11m PR curve | `yolo11m_T500_test/PR_curve.png` |")
    lines.append("")
    lines.append("## 7. Interpretation Guide")
    lines.append("")
    lines.append(
        "Precision measures how often detections are correct, while recall measures how many real objects are recovered. "
        "mAP50 evaluates detection performance at IoU = 0.50, whereas mAP50-95 evaluates stricter localization quality "
        "across multiple IoU thresholds."
    )
    lines.append("")
    lines.append(
        "If YOLO11m improves recall while reducing precision, it may be more sensitive but also more prone to false positives. "
        "If YOLOv8m maintains higher precision and similar mAP, it may remain more deployment-stable. The decision should "
        "therefore consider both accuracy and computational cost."
    )
    lines.append("")
    lines.append("## 8. Discussion Points")
    lines.append("")
    lines.append("- Which model has better recall?")
    lines.append("- Which model has better mAP50-95?")
    lines.append("- Are gains consistent across all six benchmark classes?")
    lines.append("- Which classes remain difficult?")
    lines.append("- Is YOLO11m's computational cost justified?")
    lines.append("- Are future gains more likely from model changes or data augmentation?")
    lines.append("")
    lines.append("## 9. Conclusion Template")
    lines.append("")
    lines.append(
        "The final conclusion should be written after inspecting the metrics, confusion matrices, PR curves, and rendered "
        "predictions. The results should be interpreted in the context of Report 1, where benchmark quality and class "
        "learnability were shown to be central factors in the project."
    )
    lines.append("")

    out = ROOT / "report2_T500_yolo_benchmark.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
