from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import re
import traceback
import yaml
import pandas as pd

def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def read_yaml_names(path):
    if not path or not Path(path).exists():
        return {}
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}

def metric_get(obj, path, default=None):
    cur = obj
    for part in path.split("."):
        try:
            cur = getattr(cur, part)
        except Exception:
            try:
                cur = cur[part]
            except Exception:
                return default
    try:
        return float(cur)
    except Exception:
        return cur

def read_results_csv(run_dir):
    csv = run_dir / "results.csv"
    if not csv.exists():
        return pd.DataFrame()
    return pd.read_csv(csv)

def choose_best_epoch(df, task):
    if df.empty:
        return None, None, None
    cols = list(df.columns)
    metric_candidates = []
    if task == "detect":
        metric_candidates = [c for c in cols if "mAP50-95" in c or "map50-95" in c.lower()]
        if not metric_candidates:
            metric_candidates = [c for c in cols if "mAP50" in c or "map50" in c.lower()]
    else:
        metric_candidates = [c for c in cols if "top1" in c.lower() or "accuracy" in c.lower()]
    if not metric_candidates:
        return int(df.index[-1]), None, None
    mcol = metric_candidates[0]
    vals = pd.to_numeric(df[mcol], errors="coerce")
    if vals.notna().sum() == 0:
        return int(df.index[-1]), mcol, None
    idx = int(vals.idxmax())
    epoch = int(df.loc[idx, "epoch"]) if "epoch" in df.columns else idx + 1
    return epoch, mcol, safe_float(df.loc[idx, mcol])

def final_lr(df):
    if df.empty:
        return {}
    row = df.iloc[-1]
    out = {}
    for c in df.columns:
        if "lr/" in c.lower() or c.lower().startswith("lr"):
            out[c.strip()] = safe_float(row[c])
    return out

def parse_gflops_from_log(log_path):
    if not log_path or not Path(log_path).exists():
        return None
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    vals = re.findall(r"([\d.]+)\s+GFLOPs", text)
    if not vals:
        return None
    return safe_float(vals[-1])

def load_timing(log_dir, exp_id):
    path = log_dir / f"{exp_id}_timing.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def model_info(model_path):
    out = {"model_size_MB": None, "parameters": None, "trainable_parameters": None}
    p = Path(model_path)
    if p.exists():
        out["model_size_MB"] = p.stat().st_size / (1024 * 1024)
        try:
            from ultralytics import YOLO
            model = YOLO(str(p))
            params = sum(x.numel() for x in model.model.parameters())
            trainable = sum(x.numel() for x in model.model.parameters() if x.requires_grad)
            out["parameters"] = int(params)
            out["trainable_parameters"] = int(trainable)
        except Exception as e:
            out["model_info_error"] = str(e)
    return out

def extract_detection_per_class(res, names, exp_id, dataset):
    rows = []
    box = getattr(res, "box", None)
    if box is None:
        return rows

    ap50 = getattr(box, "ap50", None)
    ap = getattr(box, "ap", None)
    ap_class_index = getattr(box, "ap_class_index", None)

    try:
        ap50 = ap50.tolist() if hasattr(ap50, "tolist") else list(ap50)
    except Exception:
        ap50 = []
    try:
        ap = ap.tolist() if hasattr(ap, "tolist") else list(ap)
    except Exception:
        ap = []
    try:
        ap_class_index = ap_class_index.tolist() if hasattr(ap_class_index, "tolist") else list(ap_class_index)
    except Exception:
        ap_class_index = list(range(max(len(ap50), len(ap))))

    for i, cid in enumerate(ap_class_index):
        cid = int(cid)
        rows.append({
            "id": exp_id,
            "dataset": dataset,
            "class_id": cid,
            "class_name": names.get(cid, str(cid)),
            "AP50": safe_float(ap50[i]) if i < len(ap50) else None,
            "AP50_95": safe_float(ap[i]) if i < len(ap) else None,
        })
    return rows

def eval_model(model_path, task, data, imgsz, device, project, name, names):
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    res = model.val(
        data=str(data),
        split="test",
        imgsz=int(imgsz),
        device=str(device),
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        save_json=True
    )

    metrics = {}
    if task == "detect":
        metrics["precision"] = metric_get(res, "box.mp")
        metrics["recall"] = metric_get(res, "box.mr")
        metrics["mAP50"] = metric_get(res, "box.map50")
        metrics["mAP50_95"] = metric_get(res, "box.map")
        per_class = extract_detection_per_class(res, names, name.replace("_test", ""), "")
    else:
        metrics["top1_acc"] = metric_get(res, "top1")
        metrics["top5_acc"] = metric_get(res, "top5")
        per_class = []

    speed = getattr(res, "speed", {}) or {}
    for k, v in speed.items():
        metrics[f"speed_{k}_ms"] = safe_float(v)

    metrics["fitness"] = metric_get(res, "fitness")
    return metrics, per_class

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--data_a_yaml", required=True)
    ap.add_argument("--data_b1", required=True)
    ap.add_argument("--data_b2_yaml", required=True)
    ap.add_argument("--imgsz", required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    eval_root = run_root / "eval"
    json_root = run_root / "eval_json"
    summary_root = run_root / "summary_tables"
    log_dir = run_root / "logs"
    for p in [eval_root, json_root, summary_root]:
        p.mkdir(parents=True, exist_ok=True)

    names_a = read_yaml_names(args.data_a_yaml)
    names_b2 = read_yaml_names(args.data_b2_yaml)

    experiments = [
        {"id": "A_yolov8m", "dataset": "A_Detection6", "task": "detect", "train_name": "A_Detection6_yolov8m_noaug", "data": args.data_a_yaml, "names": names_a},
        {"id": "A_yolo11m", "dataset": "A_Detection6", "task": "detect", "train_name": "A_Detection6_yolo11m_noaug", "data": args.data_a_yaml, "names": names_a},
        {"id": "B2_yolov8m", "dataset": "B2_Detection4", "task": "detect", "train_name": "B2_Detection4_yolov8m_noaug", "data": args.data_b2_yaml, "names": names_b2},
        {"id": "B2_yolo11m", "dataset": "B2_Detection4", "task": "detect", "train_name": "B2_Detection4_yolo11m_noaug", "data": args.data_b2_yaml, "names": names_b2},
        {"id": "B1_yolov8m_cls", "dataset": "B1_Classification", "task": "classify", "train_name": "B1_Classification_yolov8m_cls_noaug", "data": args.data_b1, "names": {}},
        {"id": "B1_yolo11m_cls", "dataset": "B1_Classification", "task": "classify", "train_name": "B1_Classification_yolo11m_cls_noaug", "data": args.data_b1, "names": {}},
    ]

    unified_rows = []
    train_rows = []
    per_class_rows = []

    for exp in experiments:
        exp_id = exp["id"]
        train_dir = run_root / "train" / exp["train_name"]
        best = train_dir / "weights" / "best.pt"
        results = read_results_csv(train_dir)
        timing = load_timing(log_dir, exp_id)
        best_epoch, best_metric_name, best_metric_value = choose_best_epoch(results, exp["task"])
        lr = final_lr(results)
        info = model_info(best)
        log_path = timing.get("log_path", str(log_dir / f"{exp_id}_train_console.log"))
        gflops = parse_gflops_from_log(log_path)
        if gflops is not None:
            info["GFLOPs_from_log"] = gflops

        epochs_completed = int(len(results)) if not results.empty else None
        elapsed_seconds = timing.get("elapsed_seconds")
        avg_epoch_seconds = elapsed_seconds / epochs_completed if elapsed_seconds and epochs_completed else None

        train_row = {
            "id": exp_id,
            "dataset": exp["dataset"],
            "task": exp["task"],
            "train_dir": str(train_dir),
            "epochs_completed": epochs_completed,
            "total_training_seconds": elapsed_seconds,
            "total_training_hours": timing.get("elapsed_hours"),
            "avg_epoch_seconds": avg_epoch_seconds,
            "best_epoch": best_epoch,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "final_lr_pg0": lr.get("lr/pg0"),
            "final_lr_pg1": lr.get("lr/pg1"),
            "final_lr_pg2": lr.get("lr/pg2"),
            **info,
        }
        train_rows.append(train_row)

        eval_name = exp_id + "_test"
        row = {
            "id": exp_id,
            "dataset": exp["dataset"],
            "task": exp["task"],
            "model_path": str(best),
            "status": "missing_model",
            **train_row,
        }

        if best.exists():
            try:
                metrics, per_class = eval_model(best, exp["task"], exp["data"], args.imgsz, args.device, eval_root, eval_name, exp["names"])
                row.update(metrics)
                row["status"] = "ok"
                eval_dir = eval_root / eval_name
                row["confusion_matrix"] = str(eval_dir / "confusion_matrix.png")
                row["confusion_matrix_normalized"] = str(eval_dir / "confusion_matrix_normalized.png")
                for pc in per_class:
                    pc["dataset"] = exp["dataset"]
                    per_class_rows.append(pc)
            except Exception as e:
                row["status"] = "eval_error"
                row["error"] = str(e)
                row["traceback"] = traceback.format_exc()

        unified_rows.append(row)
        (json_root / f"{exp_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    unified = pd.DataFrame(unified_rows)
    train = pd.DataFrame(train_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    unified.to_csv(summary_root / "unified_metrics_enhanced.csv", index=False)
    train.to_csv(summary_root / "training_efficiency_summary.csv", index=False)
    per_class_df.to_csv(summary_root / "per_class_detection_metrics.csv", index=False)

    # Compact tables for quick review
    cols = [c for c in ["id", "dataset", "task", "status", "precision", "recall", "mAP50", "mAP50_95", "top1_acc", "top5_acc", "speed_inference_ms", "total_training_hours", "avg_epoch_seconds", "best_epoch", "model_size_MB", "parameters", "GFLOPs_from_log"] if c in unified.columns]
    unified[cols].to_csv(summary_root / "unified_metrics_compact.csv", index=False)

    print(unified[cols].to_string(index=False))
    print(f"Saved enhanced tables to: {summary_root}")

if __name__ == "__main__":
    main()
