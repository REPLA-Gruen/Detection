from pathlib import Path
import argparse
import pandas as pd

def md_table(df):
    if df is None or len(df) == 0:
        return "_No data available._"
    return df.to_markdown(index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root)
    summary_root = run_root / "summary_tables"

    compact = pd.read_csv(summary_root / "unified_metrics_compact.csv") if (summary_root / "unified_metrics_compact.csv").exists() else pd.DataFrame()
    full = pd.read_csv(summary_root / "unified_metrics_enhanced.csv") if (summary_root / "unified_metrics_enhanced.csv").exists() else pd.DataFrame()
    train = pd.read_csv(summary_root / "training_efficiency_summary.csv") if (summary_root / "training_efficiency_summary.csv").exists() else pd.DataFrame()
    per_class = pd.read_csv(summary_root / "per_class_detection_metrics.csv") if (summary_root / "per_class_detection_metrics.csv").exists() else pd.DataFrame()

    lines = []
    lines.append("# Report 2 Draft — No-Augmentation YOLO Benchmark")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("This report compares YOLOv8m and YOLO11m on the final benchmark datasets developed in Report 1 under a strict no-augmentation baseline.")
    lines.append("")
    lines.append("## 2. Experimental Design")
    lines.append("| Dataset | Task | Models |")
    lines.append("|---|---|---|")
    lines.append("| Dataset A | Detection-6 | YOLOv8m, YOLO11m |")
    lines.append("| Dataset B1 | Wagon/Container classification | YOLOv8m-cls, YOLO11m-cls |")
    lines.append("| Dataset B2 | Detection-4 | YOLOv8m, YOLO11m |")
    lines.append("")
    lines.append("## 3. Unified Compact Metrics")
    lines.append(md_table(compact))
    lines.append("")
    lines.append("## 4. Training Efficiency Summary")
    show_cols = [c for c in ["id", "dataset", "task", "epochs_completed", "total_training_hours", "avg_epoch_seconds", "best_epoch", "best_metric_name", "best_metric_value", "model_size_MB", "parameters", "GFLOPs_from_log"] if c in train.columns]
    lines.append(md_table(train[show_cols] if len(train) else train))
    lines.append("")
    lines.append("## 5. Per-Class Detection Metrics")
    lines.append(md_table(per_class))
    lines.append("")
    lines.append("## 6. Figures to Insert")
    lines.append("- Confusion matrices from `eval/*/confusion_matrix.png`")
    lines.append("- Normalized confusion matrices from `eval/*/confusion_matrix_normalized.png`")
    lines.append("- PR curves and F1 curves from the evaluation folders")
    lines.append("- Training curves from each training run folder")
    lines.append("")
    lines.append("## 7. Interpretation Placeholder")
    lines.append("Complete this section after reviewing the numerical tables and confusion matrices.")
    lines.append("")
    out = run_root / "report2_draft_noaug_enhanced.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
