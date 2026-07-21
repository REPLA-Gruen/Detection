#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified evaluation for YOLOv8m vs YOLO11m on final T500 benchmark.

Outputs:
- comparison_overall.csv
- comparison_per_class_map50_95.csv
- comparison_summary.md
- metrics_summary.json per model
- per_class_map50_95.csv per model
- confusion matrices and PR curves from Ultralytics
- rendered predictions for test split
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


CONFIG_PATH = Path(__file__).with_name("benchmark_config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()

DATA_YAML = CFG["dataset_yaml"]
TEST_SOURCE = CFG["test_images"]
ROOT_OUT = Path(CFG["output_root"]) / "comparison_eval"

MODEL_V8 = CFG["yolov8_weights"]
MODEL_Y11 = CFG["yolo11_weights"]

IMGSZ = int(CFG["imgsz"])
BATCH = int(CFG["batch"])
DEVICE = int(CFG["device"])
CONF = float(CFG["conf"])
IOU = float(CFG["iou"])
CLASS_NAMES = CFG["class_names"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_overall_metrics(metrics, model_name, model_path):
    rd = getattr(metrics, "results_dict", {}) or {}
    out = {
        "model": model_name,
        "weights": model_path,
        "precision_B": None,
        "recall_B": None,
        "mAP50_B": None,
        "mAP50_95_B": None,
        "fitness": None,
    }

    for key, out_key in [
        ("metrics/precision(B)", "precision_B"),
        ("metrics/recall(B)", "recall_B"),
        ("metrics/mAP50(B)", "mAP50_B"),
        ("metrics/mAP50-95(B)", "mAP50_95_B"),
    ]:
        try:
            out[out_key] = float(rd.get(key, None))
        except Exception:
            pass

    try:
        out["fitness"] = float(metrics.fitness)
    except Exception:
        pass

    return out


def extract_speed(metrics):
    out = {
        "speed_preprocess_ms": None,
        "speed_inference_ms": None,
        "speed_loss_ms": None,
        "speed_postprocess_ms": None,
    }
    try:
        sp = getattr(metrics, "speed", {}) or {}
        out["speed_preprocess_ms"] = sp.get("preprocess", None)
        out["speed_inference_ms"] = sp.get("inference", None)
        out["speed_loss_ms"] = sp.get("loss", None)
        out["speed_postprocess_ms"] = sp.get("postprocess", None)
    except Exception:
        pass
    return out


def extract_per_class(metrics, model_name):
    rows = []
    try:
        maps = list(metrics.box.maps)
    except Exception:
        maps = None

    for i, cls_name in enumerate(CLASS_NAMES):
        rows.append({
            "model": model_name,
            "class_id": i,
            "class_name": cls_name,
            "map50_95": maps[i] if maps is not None and i < len(maps) else None,
        })

    return pd.DataFrame(rows)


def model_info(model):
    info = {
        "n_params": None,
        "n_modules": None,
        "model_task": getattr(model, "task", None),
    }

    try:
        info["n_params"] = int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        pass

    try:
        info["n_modules"] = int(len(list(model.model.modules())))
    except Exception:
        pass

    return info


def run_val(model_path: str, run_name: str):
    out_dir = ROOT_OUT / run_name
    ensure_dir(out_dir)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(model_path)

    print(f"\nRunning validation for {run_name}")
    t0 = time.time()

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        plots=True,
        project=str(ROOT_OUT),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - t0

    summary = extract_overall_metrics(metrics, run_name, model_path)
    summary["validation_wall_time_sec"] = float(elapsed)
    summary.update(extract_speed(metrics))
    summary.update(model_info(model))

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pc = extract_per_class(metrics, run_name)
    pc.to_csv(out_dir / "per_class_map50_95.csv", index=False)

    return summary, pc


def run_predict(model_path: str, run_name: str):
    pred_name = f"{run_name}_pred"
    model = YOLO(model_path)

    print(f"\nRendering predictions for {run_name}")

    model.predict(
        source=TEST_SOURCE,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(ROOT_OUT),
        name=pred_name,
        exist_ok=True,
        verbose=True,
    )


def save_markdown(overall_df, per_class_df):
    lines = []
    lines.append("# YOLOv8m vs YOLO11m — Final T500 Benchmark")
    lines.append("")
    lines.append("## Evaluation settings")
    lines.append("")
    lines.append(f"- Dataset: `{DATA_YAML}`")
    lines.append("- Split: `test`")
    lines.append(f"- Image size: `{IMGSZ}`")
    lines.append(f"- Confidence threshold: `{CONF}`")
    lines.append(f"- IoU threshold: `{IOU}`")
    lines.append(f"- Classes: `{len(CLASS_NAMES)}`")
    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    lines.append(overall_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Per-class mAP50-95")
    lines.append("")
    pivot = per_class_df.pivot(index="class_name", columns="model", values="map50_95").reset_index()
    lines.append(pivot.to_markdown(index=False))
    lines.append("")
    lines.append("## Figure folders")
    lines.append("")
    lines.append("- YOLOv8m validation figures: `yolov8m_T500_test`")
    lines.append("- YOLO11m validation figures: `yolo11m_T500_test`")
    lines.append("- YOLOv8m rendered predictions: `yolov8m_T500_test_pred`")
    lines.append("- YOLO11m rendered predictions: `yolo11m_T500_test_pred`")
    lines.append("")

    (ROOT_OUT / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dir(ROOT_OUT)

    v8_summary, v8_pc = run_val(MODEL_V8, "yolov8m_T500_test")
    run_predict(MODEL_V8, "yolov8m_T500_test")

    y11_summary, y11_pc = run_val(MODEL_Y11, "yolo11m_T500_test")
    run_predict(MODEL_Y11, "yolo11m_T500_test")

    overall_df = pd.DataFrame([v8_summary, y11_summary])
    overall_df.to_csv(ROOT_OUT / "comparison_overall.csv", index=False)

    per_class_df = pd.concat([v8_pc, y11_pc], ignore_index=True)
    per_class_df.to_csv(ROOT_OUT / "comparison_per_class_map50_95.csv", index=False)

    save_markdown(overall_df, per_class_df)

    print("\nDone.")
    print(f"Results saved to: {ROOT_OUT}")


if __name__ == "__main__":
    main()
