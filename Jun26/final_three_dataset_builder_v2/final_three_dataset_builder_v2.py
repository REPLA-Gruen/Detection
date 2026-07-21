#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DET6_NAMES = {
    0: "BAW-Sack",
    1: "Plastic",
    2: "Paper_Cardboard",
    3: "Wagon",
    4: "Container",
    5: "Other",
}

DET4_NAMES = {
    0: "BAW-Sack",
    1: "Plastic",
    2: "Paper_Cardboard",
    3: "Other",
}

GROUP_BY_NAME_DET6 = {
    "BAW-Sack": 0,
    "Plastiksack": 1,
    "PET 0.5l": 1,
    "PET 1.5l": 1,
    "Styropor/Dämmung": 1,
    "Blumentopf": 1,
    "Gummiband": 1,
    "Papier": 2,
    "Karton": 2,
    "Wagon": 3,
    "Container": 4,
}

GROUP_BY_NAME_DET4 = {
    "BAW-Sack": 0,
    "Plastiksack": 1,
    "PET 0.5l": 1,
    "PET 1.5l": 1,
    "Styropor/Dämmung": 1,
    "Blumentopf": 1,
    "Gummiband": 1,
    "Papier": 2,
    "Karton": 2,
}

CLASSIFICATION_CLASSES = ["Wagon", "Container"]


def read_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def find_images(root: Path):
    return sorted([p for p in (root / "images").rglob("*") if p.suffix.lower() in IMG_EXTS])


def label_for_image(img: Path, root: Path):
    rel = img.relative_to(root / "images")
    return (root / "labels" / rel).with_suffix(".txt")


def parse_label(label: Path):
    rows = []
    if not label.exists():
        return rows
    text = label.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            coords = parts[1:]
            rows.append((cid, coords))
        except Exception:
            continue
    return rows


def safe_clear(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_detection_yaml(out_root: Path, names: dict):
    cfg = {
        "path": str(out_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def make_item(img: Path, raw_root: Path, names: dict):
    label = label_for_image(img, raw_root)
    objs = parse_label(label)
    raw_names = [names[cid] for cid, _ in objs if cid in names]

    has_wagon = "Wagon" in raw_names
    has_container = "Container" in raw_names

    if has_wagon and not has_container:
        context_class = "Wagon"
    elif has_container and not has_wagon:
        context_class = "Container"
    else:
        context_class = None

    det6_classes = []
    det4_classes = []

    for cname in raw_names:
        det6_classes.append(GROUP_BY_NAME_DET6.get(cname, 5))
        if cname in ["Wagon", "Container"]:
            continue
        det4_classes.append(GROUP_BY_NAME_DET4.get(cname, 3))

    rel_image = img.relative_to(raw_root / "images")

    return {
        "image": img,
        "label": label,
        "rel_image": rel_image,
        "rel_id": str(rel_image).replace("\\", "/"),
        "objs": objs,
        "raw_names": raw_names,
        "has_wagon": has_wagon,
        "has_container": has_container,
        "has_both": has_wagon and has_container,
        "has_neither": (not has_wagon and not has_container),
        "context_class": context_class,
        "include_in_universe": context_class in CLASSIFICATION_CLASSES,
        "det6_classes": det6_classes,
        "det4_classes": det4_classes,
    }


def build_items(raw_root: Path, names: dict):
    return [make_item(img, raw_root, names) for img in find_images(raw_root)]


def split_by_buckets(items, class_key, ratios, seed, mode, n_trials=5000):
    """
    Repeated randomized bucketed split.
    mode:
      detection: classes are object class lists
      classification: class_key is context_class
    """
    rng = np.random.default_rng(seed)
    n = len(items)
    target_images = {
        "train": int(round(ratios["train"] * n)),
        "val": int(round(ratios["val"] * n)),
        "test": n - int(round(ratios["train"] * n)) - int(round(ratios["val"] * n)),
    }

    best = None
    best_score = None
    best_detail = None

    for trial in range(n_trials):
        split = {"train": [], "val": [], "test": []}

        buckets = defaultdict(list)
        for idx, it in enumerate(items):
            if mode == "classification":
                sig = it["context_class"]
            else:
                classes = it[class_key]
                sig = tuple(sorted(set(classes)))
            buckets[sig].append(idx)

        # Randomize bucket order too
        bucket_keys = list(buckets.keys())
        rng.shuffle(bucket_keys)

        for sig in bucket_keys:
            idxs = list(buckets[sig])
            rng.shuffle(idxs)
            m = len(idxs)
            n_train = int(round(ratios["train"] * m))
            n_val = int(round(ratios["val"] * m))
            n_test = m - n_train - n_val

            split["train"].extend(idxs[:n_train])
            split["val"].extend(idxs[n_train:n_train+n_val])
            split["test"].extend(idxs[n_train+n_val:])

        # rebalance exact image counts by moving random images if needed
        split = rebalance_image_counts(split, target_images, rng)

        score, detail = score_split(items, split, class_key, ratios, mode)
        if best_score is None or score < best_score:
            best = {k: list(v) for k, v in split.items()}
            best_score = score
            best_detail = detail

    return best, best_score, best_detail


def rebalance_image_counts(split, target_images, rng):
    # Move images from oversized split to undersized split.
    changed = True
    while changed:
        changed = False
        over = [s for s in ["train", "val", "test"] if len(split[s]) > target_images[s]]
        under = [s for s in ["train", "val", "test"] if len(split[s]) < target_images[s]]
        if over and under:
            s_from = over[0]
            s_to = under[0]
            idx_pos = int(rng.integers(0, len(split[s_from])))
            idx = split[s_from].pop(idx_pos)
            split[s_to].append(idx)
            changed = True
    return split


def score_split(items, split, class_key, ratios, mode):
    score = 0.0
    detail = {}

    # image ratio penalty
    n = len(items)
    image_penalty = 0
    for s in ["train", "val", "test"]:
        image_penalty += abs(len(split[s]) / max(n, 1) - ratios[s])
    score += image_penalty * 10

    if mode == "classification":
        all_counts = Counter([it["context_class"] for it in items])
        split_counts = {}
        missing = 0
        ratio_penalty = 0
        for s in ["train", "val", "test"]:
            c = Counter([items[i]["context_class"] for i in split[s]])
            split_counts[s] = c
            for cls in CLASSIFICATION_CLASSES:
                if c[cls] == 0:
                    missing += 1
                ratio_penalty += abs(c[cls] / max(all_counts[cls], 1) - ratios[s])
        score += missing * 1_000_000 + ratio_penalty
        detail = {
            "missing": missing,
            "ratio_penalty": ratio_penalty,
            "image_penalty": image_penalty,
        }
    else:
        all_counts = Counter()
        for it in items:
            all_counts.update(it[class_key])
        class_ids = sorted(all_counts.keys())
        missing = 0
        ratio_penalty = 0
        density = []
        for s in ["train", "val", "test"]:
            c = Counter()
            n_objects = 0
            for i in split[s]:
                c.update(items[i][class_key])
                n_objects += len(items[i][class_key])
            density.append(n_objects / max(len(split[s]), 1))
            for cls in class_ids:
                if all_counts[cls] > 0 and c[cls] == 0:
                    missing += 1
                ratio_penalty += abs(c[cls] / max(all_counts[cls], 1) - ratios[s])
        density_penalty = float(np.std(density) / max(np.mean(density), 1e-9))
        score += missing * 1_000_000 + ratio_penalty + density_penalty
        detail = {
            "missing": missing,
            "ratio_penalty": ratio_penalty,
            "density_penalty": density_penalty,
            "image_penalty": image_penalty,
        }

    return float(score), detail


def convert_detection_label(objs, names, mode):
    out = []
    for cid, coords in objs:
        cname = names.get(cid)
        if mode == "det6":
            new_id = GROUP_BY_NAME_DET6.get(cname, 5)
        elif mode == "det4":
            if cname in ["Wagon", "Container"]:
                continue
            new_id = GROUP_BY_NAME_DET4.get(cname, 3)
        else:
            raise ValueError(mode)
        out.append(" ".join([str(new_id)] + coords))
    return "\n".join(out) + ("\n" if out else "")


def copy_detection_dataset(items, split, out_root, names, mode):
    rows = []
    for split_name, idxs in split.items():
        for idx in idxs:
            it = items[idx]
            img = it["image"]
            rel = it["rel_image"]
            out_img = out_root / "images" / split_name / rel.name
            out_lbl = out_root / "labels" / split_name / rel.with_suffix(".txt").name
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_img)
            text = convert_detection_label(it["objs"], names, mode)
            out_lbl.write_text(text, encoding="utf-8")
            rows.append({
                "split": split_name,
                "rel_id": it["rel_id"],
                "image": str(out_img),
                "label": str(out_lbl),
                "n_objects": len(text.strip().splitlines()) if text.strip() else 0,
            })
    return pd.DataFrame(rows)


def copy_classification_dataset(items, split, out_root):
    rows = []
    for split_name, idxs in split.items():
        for idx in idxs:
            it = items[idx]
            cls = it["context_class"]
            img = it["image"]
            rel = it["rel_image"]
            out_img = out_root / split_name / cls / rel.name
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_img)
            rows.append({
                "split": split_name,
                "class": cls,
                "rel_id": it["rel_id"],
                "image": str(out_img),
            })
    return pd.DataFrame(rows)


def summarize_detection(out_root: Path, names: dict):
    class_rows = []
    split_rows = []
    for split_name in ["train", "val", "test"]:
        labels = sorted((out_root / "labels" / split_name).rglob("*.txt"))
        counter = Counter()
        n_objects = 0
        empty = 0
        for lbl in labels:
            text = lbl.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                empty += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                if not parts:
                    continue
                cid = int(float(parts[0]))
                counter[cid] += 1
                n_objects += 1
        split_rows.append({
            "split": split_name,
            "images": len(labels),
            "objects": n_objects,
            "objects_per_image": n_objects / max(len(labels), 1),
            "empty_labels": empty,
        })
        for cid, cname in names.items():
            class_rows.append({
                "split": split_name,
                "class_id": cid,
                "class_name": cname,
                "object_count": int(counter[cid]),
            })
    return pd.DataFrame(class_rows), pd.DataFrame(split_rows)


def summarize_classification(out_root: Path):
    rows = []
    split_rows = []
    for split_name in ["train", "val", "test"]:
        total = 0
        for cls in CLASSIFICATION_CLASSES:
            n = len([p for p in (out_root / split_name / cls).glob("*") if p.is_file()])
            total += n
            rows.append({"split": split_name, "class": cls, "n_images": n})
        split_rows.append({"split": split_name, "images": total})
    return pd.DataFrame(rows), pd.DataFrame(split_rows)


def pivot_detection(dist):
    return dist.pivot(index="class_name", columns="split", values="object_count").reset_index()


def pivot_classification(dist):
    return dist.pivot(index="class", columns="split", values="n_images").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--yaml", required=True, type=Path)
    ap.add_argument("--out_a", required=True, type=Path)
    ap.add_argument("--out_b1", required=True, type=Path)
    ap.add_argument("--out_b2", required=True, type=Path)
    ap.add_argument("--report_dir", required=True, type=Path)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_trials", type=int, default=5000)
    ap.add_argument("--clear_outputs", action="store_true")
    args = ap.parse_args()

    ratios = {"train": args.train, "val": args.val, "test": args.test}

    if args.clear_outputs:
        for p in [args.out_a, args.out_b1, args.out_b2, args.report_dir]:
            safe_clear(p)
    else:
        for p in [args.out_a, args.out_b1, args.out_b2, args.report_dir]:
            p.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(args.yaml)
    names = normalize_names(cfg["names"])

    all_items = build_items(args.dataset_root, names)
    universe_items = [it for it in all_items if it["include_in_universe"]]
    excluded_items = [it for it in all_items if not it["include_in_universe"]]

    # Independent task-specific optimized splits
    split_a, score_a, detail_a = split_by_buckets(universe_items, "det6_classes", ratios, args.seed + 1, "detection", args.n_trials)
    split_b1, score_b1, detail_b1 = split_by_buckets(universe_items, "context_class", ratios, args.seed + 2, "classification", args.n_trials)
    split_b2, score_b2, detail_b2 = split_by_buckets(universe_items, "det4_classes", ratios, args.seed + 3, "detection", args.n_trials)

    # Copy datasets
    df_a = copy_detection_dataset(universe_items, split_a, args.out_a, names, "det6")
    df_b1 = copy_classification_dataset(universe_items, split_b1, args.out_b1)
    df_b2 = copy_detection_dataset(universe_items, split_b2, args.out_b2, names, "det4")

    write_detection_yaml(args.out_a, DET6_NAMES)
    write_detection_yaml(args.out_b2, DET4_NAMES)

    df_a.to_csv(args.report_dir / "dataset_A_detection6_plan.csv", index=False)
    df_b1.to_csv(args.report_dir / "dataset_B1_classification_plan.csv", index=False)
    df_b2.to_csv(args.report_dir / "dataset_B2_detection4_plan.csv", index=False)

    excluded_df = pd.DataFrame([{
        "rel_id": it["rel_id"],
        "image": str(it["image"]),
        "has_wagon": it["has_wagon"],
        "has_container": it["has_container"],
        "has_both": it["has_both"],
        "has_neither": it["has_neither"],
        "raw_names": ",".join(it["raw_names"]),
    } for it in excluded_items])
    excluded_df.to_csv(args.report_dir / "excluded_images.csv", index=False)

    universe_df = pd.DataFrame([{
        "rel_id": it["rel_id"],
        "context_class": it["context_class"],
        "det6_classes": ",".join(map(str, sorted(set(it["det6_classes"])))),
        "det4_classes": ",".join(map(str, sorted(set(it["det4_classes"])))),
    } for it in universe_items])
    universe_df.to_csv(args.report_dir / "benchmark_universe.csv", index=False)

    # Split IDs per dataset
    for dataset_name, split in [("A", split_a), ("B1", split_b1), ("B2", split_b2)]:
        for split_name, idxs in split.items():
            pd.DataFrame([{
                "rel_id": universe_items[idx]["rel_id"],
                "context_class": universe_items[idx]["context_class"],
            } for idx in idxs]).to_csv(args.report_dir / f"{dataset_name}_{split_name}_ids.csv", index=False)

    a_dist, a_split = summarize_detection(args.out_a, DET6_NAMES)
    b1_dist, b1_split = summarize_classification(args.out_b1)
    b2_dist, b2_split = summarize_detection(args.out_b2, DET4_NAMES)

    a_dist.to_csv(args.report_dir / "dataset_A_detection6_distribution.csv", index=False)
    a_split.to_csv(args.report_dir / "dataset_A_detection6_split_summary.csv", index=False)
    b1_dist.to_csv(args.report_dir / "dataset_B1_classification_distribution.csv", index=False)
    b1_split.to_csv(args.report_dir / "dataset_B1_classification_split_summary.csv", index=False)
    b2_dist.to_csv(args.report_dir / "dataset_B2_detection4_distribution.csv", index=False)
    b2_split.to_csv(args.report_dir / "dataset_B2_detection4_split_summary.csv", index=False)

    pivot_detection(a_dist).to_csv(args.report_dir / "dataset_A_detection6_distribution_pivot.csv", index=False)
    pivot_classification(b1_dist).to_csv(args.report_dir / "dataset_B1_classification_distribution_pivot.csv", index=False)
    pivot_detection(b2_dist).to_csv(args.report_dir / "dataset_B2_detection4_distribution_pivot.csv", index=False)

    split_quality = pd.DataFrame([
        {"dataset": "A_Detection6", "score": score_a, **detail_a},
        {"dataset": "B1_Classification_WC", "score": score_b1, **detail_b1},
        {"dataset": "B2_Detection4", "score": score_b2, **detail_b2},
    ])
    split_quality.to_csv(args.report_dir / "split_quality_scores.csv", index=False)

    quality_rows = [
        {"check": "same_universe_size_all_datasets", "status": len(df_a) == len(df_b1) == len(df_b2), "value": f"A={len(df_a)}, B1={len(df_b1)}, B2={len(df_b2)}"},
        {"check": "excluded_images", "status": len(excluded_items) == 25, "value": len(excluded_items)},
        {"check": "benchmark_universe", "status": len(universe_items) == 3334, "value": len(universe_items)},
        {"check": "A_has_6_classes", "status": a_dist["class_name"].nunique() == 6, "value": a_dist["class_name"].nunique()},
        {"check": "B1_has_2_classes", "status": b1_dist["class"].nunique() == 2, "value": b1_dist["class"].nunique()},
        {"check": "B2_has_4_classes", "status": b2_dist["class_name"].nunique() == 4, "value": b2_dist["class_name"].nunique()},
    ]
    qc = pd.DataFrame(quality_rows)
    qc.to_csv(args.report_dir / "quality_checks.csv", index=False)

    raw_summary = pd.DataFrame([
        {"metric": "raw_corrected_images", "value": len(all_items)},
        {"metric": "raw_corrected_objects", "value": sum(len(it["objs"]) for it in all_items)},
        {"metric": "excluded_images", "value": len(excluded_items)},
        {"metric": "benchmark_universe_images", "value": len(universe_items)},
    ])
    raw_summary.to_csv(args.report_dir / "raw_corrected_summary.csv", index=False)

    final_summary = pd.DataFrame([
        {"metric": "dataset_A_images", "value": len(df_a)},
        {"metric": "dataset_A_objects", "value": int(a_split["objects"].sum())},
        {"metric": "dataset_B1_images", "value": len(df_b1)},
        {"metric": "dataset_B2_images", "value": len(df_b2)},
        {"metric": "dataset_B2_objects", "value": int(b2_split["objects"].sum())},
        {"metric": "all_datasets_same_image_count", "value": len(df_a) == len(df_b1) == len(df_b2)},
        {"metric": "task_specific_splits", "value": True},
    ])
    final_summary.to_csv(args.report_dir / "final_summary.csv", index=False)
    (args.report_dir / "final_summary.json").write_text(json.dumps({r["metric"]: r["value"] for _, r in final_summary.iterrows()}, indent=2), encoding="utf-8")

    # Markdown report
    lines = []
    lines.append("# Final Three-Dataset Builder v2 Report")
    lines.append("")
    lines.append("## 1. Raw Corrected Dataset Summary")
    lines.append(raw_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 2. Final Dataset Summary")
    lines.append(final_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Quality Checks")
    lines.append(qc.to_markdown(index=False))
    lines.append("")
    lines.append("## 4. Split Quality Scores")
    lines.append(split_quality.to_markdown(index=False))
    lines.append("")
    lines.append("## 5. Dataset A — Detection-6 Split Summary")
    lines.append(a_split.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. Dataset A — Detection-6 Class Distribution")
    lines.append(pivot_detection(a_dist).to_markdown(index=False))
    lines.append("")
    lines.append("## 7. Dataset B1 — Classification Split Summary")
    lines.append(b1_split.to_markdown(index=False))
    lines.append("")
    lines.append("## 8. Dataset B1 — Classification Distribution")
    lines.append(pivot_classification(b1_dist).to_markdown(index=False))
    lines.append("")
    lines.append("## 9. Dataset B2 — Detection-4 Split Summary")
    lines.append(b2_split.to_markdown(index=False))
    lines.append("")
    lines.append("## 10. Dataset B2 — Detection-4 Class Distribution")
    lines.append(pivot_detection(b2_dist).to_markdown(index=False))
    lines.append("")
    lines.append("## 11. Excluded Images")
    lines.append(f"Excluded images: {len(excluded_items)}")
    lines.append("")
    if len(excluded_df):
        lines.append(excluded_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 12. Interpretation")
    lines.append("All three final datasets use the same 3334-image benchmark universe.")
    lines.append("The splits are optimized separately for each task to maximize class balance and evaluation stability.")
    lines.append("This design supports fair strategy-level comparison while allowing each learning task to use its most stable train/validation/test partition.")
    (args.report_dir / "report_final_three_dataset_builder_v2.md").write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(final_summary.to_string(index=False))
    print(f"Report folder: {args.report_dir}")


if __name__ == "__main__":
    main()
