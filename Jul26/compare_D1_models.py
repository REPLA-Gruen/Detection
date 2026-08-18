#!/usr/bin/env python3
"""Compare two or more D1 evaluation result folders."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "matthews_correlation_coefficient",
    "roc_auc",
    "correct",
    "incorrect",
    "wall_ms_per_image",
    "ultralytics_inference_ms_per_image",
    "parameters",
    "checkpoint_size_mb",
]


def resolve_metrics(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = [path / "reports" / "metrics.json", path / "metrics.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"metrics.json not found under: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare D1 classifier evaluation results.")
    p.add_argument("--evaluation", type=Path, nargs="+", required=True, help="Evaluation folders or metrics.json files.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        print(f"ERROR: Output exists: {output}; use --overwrite", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in args.evaluation:
        metrics_path = resolve_metrics(item)
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "model_name": data.get("model_name", metrics_path.parent.parent.name),
            "model": data.get("model", ""),
            "dataset": data.get("dataset", ""),
            "split": data.get("split", ""),
            "images": data.get("images", ""),
            "metrics_file": str(metrics_path),
        }
        for key in KEYS:
            row[key] = data.get(key, "")
        rows.append(row)

    rows.sort(key=lambda row: float(row.get("macro_f1") or -1), reverse=True)
    fields = ["rank", "model_name", "model", "dataset", "split", "images", *KEYS, "metrics_file"]
    with (output / "D1_model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})

    lines = [
        "# D1 classifier comparison",
        "",
        "Models are ranked by macro F1. Use test-set results generated with identical settings.",
        "",
        "| Rank | Model | Accuracy | Balanced accuracy | Macro F1 | MCC | ROC-AUC | ms/image | Parameters |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        def fmt(key: str, digits: int = 6) -> str:
            value = row.get(key, "")
            if value == "" or value is None:
                return ""
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return str(value)
        lines.append(
            f"| {rank} | {row['model_name']} | {fmt('accuracy')} | {fmt('balanced_accuracy')} | "
            f"{fmt('macro_f1')} | {fmt('matthews_correlation_coefficient')} | {fmt('roc_auc')} | "
            f"{fmt('ultralytics_inference_ms_per_image', 3)} | {row.get('parameters', '')} |"
        )
    (output / "D1_model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
