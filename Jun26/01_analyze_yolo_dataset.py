#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze a YOLO dataset for foreign material detection.

Supports both:
- YOLO bbox labels: class x_center y_center width height
- YOLO polygon/segmentation labels: class x1 y1 x2 y2 ...

Outputs CSVs, plots, and Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("Unsupported names format in YAML.")


def find_images(dataset_root: Path) -> list[Path]:
    img_root = dataset_root / "images"
    if not img_root.exists():
        raise FileNotFoundError(f"images folder not found: {img_root}")
    return sorted([p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def image_to_label_path(img_path: Path, dataset_root: Path) -> Path:
    rel = img_path.relative_to(dataset_root / "images")
    return (dataset_root / "labels" / rel).with_suffix(".txt")


def parse_label_file(label_path: Path):
    """
    Returns list of objects:
    dict(class_id, label_type, x_center, y_center, width, height, area_norm, n_points)
    For polygon labels, bbox is derived from polygon min/max.
    """
    objects = []
    issues = []

    if not label_path.exists():
        issues.append(("missing_label", str(label_path)))
        return objects, issues

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return objects, issues

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            issues.append(("malformed_line_too_short", f"{label_path}:{line_no}"))
            continue

        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except Exception:
            issues.append(("non_numeric_label", f"{label_path}:{line_no}"))
            continue

        if any((v < -1e-6 or v > 1 + 1e-6) for v in vals):
            issues.append(("coordinates_outside_0_1", f"{label_path}:{line_no}"))

        # YOLO bbox: exactly 4 values
        if len(vals) == 4:
            xc, yc, w, h = vals
            label_type = "bbox"
            n_points = 0
        else:
            # YOLO polygon: even number of x,y values
            if len(vals) % 2 != 0:
                issues.append(("polygon_odd_number_of_coordinates", f"{label_path}:{line_no}"))
                continue
            xs = np.array(vals[0::2], dtype=float)
            ys = np.array(vals[1::2], dtype=float)
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            xc = xmin + w / 2.0
            yc = ymin + h / 2.0
            label_type = "polygon"
            n_points = len(xs)

        area = max(0.0, w) * max(0.0, h)

        objects.append({
            "class_id": cls,
            "label_type": label_type,
            "x_center": xc,
            "y_center": yc,
            "width_norm": w,
            "height_norm": h,
            "area_norm": area,
            "n_points": n_points,
        })

    return objects, issues


def object_size_category(area_norm: float):
    # COCO-like normalized approximation, adjusted for YOLO normalized bbox area
    if area_norm < 0.01:
        return "small"
    if area_norm < 0.10:
        return "medium"
    return "large"


def analyze(dataset_root: Path, yaml_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    cfg = read_yaml(yaml_path)
    names = normalize_names(cfg["names"])
    nc = int(cfg.get("nc", len(names)))

    images = find_images(dataset_root)

    image_rows = []
    object_rows = []
    issue_rows = []

    for img_path in tqdm(images, desc="Analyzing images"):
        rel_img = str(img_path.relative_to(dataset_root))
        label_path = image_to_label_path(img_path, dataset_root)

        try:
            with Image.open(img_path) as im:
                w_px, h_px = im.size
        except Exception as e:
            issue_rows.append({"issue": "image_read_error", "path": rel_img, "detail": str(e)})
            w_px, h_px = None, None

        objects, issues = parse_label_file(label_path)

        for issue, detail in issues:
            issue_rows.append({"issue": issue, "path": rel_img, "detail": detail})

        classes_in_image = sorted(set([o["class_id"] for o in objects]))

        image_rows.append({
            "image": rel_img,
            "label": str(label_path.relative_to(dataset_root)) if label_path.exists() else "",
            "width_px": w_px,
            "height_px": h_px,
            "n_objects": len(objects),
            "n_classes": len(classes_in_image),
            "classes": ",".join(map(str, classes_in_image)),
            "has_label_file": label_path.exists(),
            "is_empty": len(objects) == 0,
        })

        for idx, obj in enumerate(objects):
            cls = obj["class_id"]
            cname = names.get(cls, f"UNKNOWN_{cls}")

            if cls not in names:
                issue_rows.append({"issue": "unknown_class_id", "path": rel_img, "detail": f"class_id={cls}"})

            object_rows.append({
                "image": rel_img,
                "object_index": idx,
                "class_id": cls,
                "class_name": cname,
                **obj,
                "area_px_est": obj["area_norm"] * w_px * h_px if w_px and h_px else np.nan,
                "size_category": object_size_category(obj["area_norm"]),
            })

    image_df = pd.DataFrame(image_rows)
    object_df = pd.DataFrame(object_rows)
    issues_df = pd.DataFrame(issue_rows)

    image_df.to_csv(output_dir / "image_stats.csv", index=False)
    object_df.to_csv(output_dir / "object_stats.csv", index=False)
    issues_df.to_csv(output_dir / "quality_issues.csv", index=False)

    # class distribution
    class_counts = object_df.groupby(["class_id", "class_name"]).size().reset_index(name="n_objects") if len(object_df) else pd.DataFrame(columns=["class_id","class_name","n_objects"])
    image_class_rows = []
    for cid, cname in names.items():
        if len(object_df):
            imgs_with = object_df.loc[object_df["class_id"] == cid, "image"].nunique()
        else:
            imgs_with = 0
        image_class_rows.append({"class_id": cid, "class_name": cname, "n_images_containing_class": imgs_with})
    class_img_df = pd.DataFrame(image_class_rows)

    all_classes = pd.DataFrame([{"class_id": cid, "class_name": cname} for cid, cname in names.items()])
    class_dist = all_classes.merge(class_counts, on=["class_id","class_name"], how="left").fillna({"n_objects":0})
    class_dist["n_objects"] = class_dist["n_objects"].astype(int)
    total_objects = int(class_dist["n_objects"].sum())
    class_dist["object_percent"] = class_dist["n_objects"] / max(total_objects, 1) * 100.0
    class_dist = class_dist.merge(class_img_df, on=["class_id","class_name"], how="left")
    class_dist.to_csv(output_dir / "class_distribution.csv", index=False)
    class_img_df.to_csv(output_dir / "class_image_distribution.csv", index=False)

    # bbox distribution
    if len(object_df):
        bbox_size_df = object_df.groupby(["class_id","class_name","size_category"]).size().reset_index(name="n_objects")
    else:
        bbox_size_df = pd.DataFrame(columns=["class_id","class_name","size_category","n_objects"])
    bbox_size_df.to_csv(output_dir / "bbox_size_distribution.csv", index=False)

    # summary
    n_images = len(image_df)
    n_empty = int(image_df["is_empty"].sum()) if n_images else 0
    n_label_files_missing = int((~image_df["has_label_file"]).sum()) if n_images else 0
    imbalance_ratio = None
    nonzero = class_dist[class_dist["n_objects"] > 0]["n_objects"]
    if len(nonzero) > 0:
        imbalance_ratio = float(nonzero.max() / max(nonzero.min(), 1))

    summary = {
        "dataset_root": str(dataset_root),
        "yaml": str(yaml_path),
        "nc": nc,
        "n_classes": len(names),
        "n_images": int(n_images),
        "n_objects": int(total_objects),
        "n_empty_images": n_empty,
        "empty_image_percent": n_empty / max(n_images, 1) * 100.0,
        "n_missing_label_files": n_label_files_missing,
        "objects_per_image_mean": float(image_df["n_objects"].mean()) if n_images else 0.0,
        "objects_per_image_median": float(image_df["n_objects"].median()) if n_images else 0.0,
        "imbalance_ratio_nonzero_classes": imbalance_ratio,
        "n_quality_issues": int(len(issues_df)),
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(image_df, object_df, class_dist, bbox_size_df, fig_dir)
    write_report(summary, class_dist, bbox_size_df, issues_df, output_dir / "report_dataset_analysis.md")

    print("\nDone.")
    print(f"Images: {summary['n_images']}")
    print(f"Objects: {summary['n_objects']}")
    print(f"Classes: {summary['n_classes']}")
    print(f"Output: {output_dir}")


def make_plots(image_df, object_df, class_dist, bbox_size_df, fig_dir):
    # class objects
    fig, ax = plt.subplots(figsize=(12, 6))
    tmp = class_dist.sort_values("n_objects", ascending=False)
    ax.bar(tmp["class_name"], tmp["n_objects"])
    ax.set_title("Objects per Class")
    ax.set_ylabel("Number of objects")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution_objects.png", dpi=160)
    plt.close(fig)

    # images per class
    fig, ax = plt.subplots(figsize=(12, 6))
    tmp = class_dist.sort_values("n_images_containing_class", ascending=False)
    ax.bar(tmp["class_name"], tmp["n_images_containing_class"])
    ax.set_title("Images Containing Each Class")
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution_images.png", dpi=160)
    plt.close(fig)

    # object density
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(image_df["n_objects"], bins=30)
    ax.set_title("Objects per Image")
    ax.set_xlabel("Objects per image")
    ax.set_ylabel("Image count")
    fig.tight_layout()
    fig.savefig(fig_dir / "objects_per_image_hist.png", dpi=160)
    plt.close(fig)

    if len(object_df):
        # bbox area
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(object_df["area_norm"], bins=50)
        ax.set_title("Normalized Bounding Box Area Distribution")
        ax.set_xlabel("Normalized bbox area")
        ax.set_ylabel("Object count")
        fig.tight_layout()
        fig.savefig(fig_dir / "bbox_area_distribution.png", dpi=160)
        plt.close(fig)

        # width height scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(object_df["width_norm"], object_df["height_norm"], s=8, alpha=0.4)
        ax.set_title("Normalized BBox Width vs Height")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.tight_layout()
        fig.savefig(fig_dir / "bbox_width_height_scatter.png", dpi=160)
        plt.close(fig)

        # size category
        pivot = bbox_size_df.pivot_table(index="class_name", columns="size_category", values="n_objects", fill_value=0)
        pivot = pivot.reindex(class_dist["class_name"])
        pivot.plot(kind="bar", stacked=True, figsize=(12,6))
        plt.title("Small / Medium / Large Objects per Class")
        plt.ylabel("Object count")
        plt.xticks(rotation=75)
        plt.tight_layout()
        plt.savefig(fig_dir / "object_size_category_by_class.png", dpi=160)
        plt.close()


def write_report(summary, class_dist, bbox_size_df, issues_df, out_path: Path):
    top = class_dist.sort_values("n_objects", ascending=False).head(10)
    rare = class_dist[class_dist["n_objects"] > 0].sort_values("n_objects", ascending=True).head(10)
    zero = class_dist[class_dist["n_objects"] == 0]

    lines = []
    lines.append("# Dataset Analysis Report")
    lines.append("")
    lines.append("## Summary")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Most Frequent Classes")
    lines.append(top.to_markdown(index=False))
    lines.append("")
    lines.append("## Rarest Non-Zero Classes")
    lines.append(rare.to_markdown(index=False))
    lines.append("")
    if len(zero):
        lines.append("## Classes With Zero Objects")
        lines.append(zero[["class_id","class_name"]].to_markdown(index=False))
        lines.append("")
    lines.append("## Interpretation")
    lines.append("- This dataset should be assessed for class imbalance before training.")
    lines.append("- Rare classes should be protected during train/validation/test splitting.")
    lines.append("- If rare classes remain underrepresented, synthetic augmentation or targeted data collection is recommended.")
    lines.append("")
    lines.append("## Quality Issues")
    if len(issues_df):
        lines.append(f"- Number of detected quality issues: {len(issues_df)}")
        lines.append("")
        lines.append(issues_df.head(50).to_markdown(index=False))
    else:
        lines.append("- No critical quality issues detected by the script.")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("- `figures/class_distribution_objects.png`")
    lines.append("- `figures/class_distribution_images.png`")
    lines.append("- `figures/objects_per_image_hist.png`")
    lines.append("- `figures/bbox_area_distribution.png`")
    lines.append("- `figures/bbox_width_height_scatter.png`")
    lines.append("- `figures/object_size_category_by_class.png`")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--yaml", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    analyze(args.dataset_root, args.yaml, args.output_dir)


if __name__ == "__main__":
    main()
