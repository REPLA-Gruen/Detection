# -*- coding: utf-8 -*-
"""
Summarize Label Studio annotations into 8 normalized classes ONLY and
emit a raw->normalized mapping report so the merge is fully transparent.

Outputs into --out_dir:
  - labels_summary_8.csv        (exactly 8 rows: class, instances, images, example_1)
  - raw_to_norm_counts.csv      (per raw label: normalized target, instances, images)
  - lists/<class>.txt           (image names that contain at least one instance of that class)
BAW-Sack ⇐ {BAW-Sack, BAW Sack, BAW}

Plastic ⇐ {Plastiksack, Plastic-Sack, Plastic, PET 0.5l, PET 1.5l, PET}

Metal ⇐ {Alubuechse, Kaffeekapsel, Aluminium}

Glass ⇐ {Glasflasche, Glass}

Paper_Cardboard ⇐ {Papier, Pappe, Paper, Cardboard}

Wagon ⇐ {Wagon}

Container ⇐ {Container}

Other ⇐ everything else that isn’t in the sets above (e.g., Diverses, Blumentopf, Windel, unknowns)







python C:\Hadi\Foreign_Material\Code\Python\tools\2-ls_label_stats_norm8.py `
  --export_json "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\project-2-at-2026-01-06-10-09-5c186fe2.json" `
  --images_dir  "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\images" `
  --out_dir     "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\stats_norm8" `
  --examples 1




"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict

TARGET_CLASSES = [
    "BAW-Sack", "Plastic", "Metal", "Glass",
    "Paper_Cardboard", "Other", "Wagon", "Container"
]

# ---- Raw -> Normalized merge rules (extend as needed) ----
RAW_TO_NORM: Dict[str, str] = {
    # BAW-Sack
    "BAW-Sack":"BAW-Sack", "BAW Sack":"BAW-Sack", "BAW":"BAW-Sack",

    # Plastic
    "Plastiksack":"Plastic", "Plastic-Sack":"Plastic", "Plastic":"Plastic",
    "PET 0.5l":"Plastic", "PET 1.5l":"Plastic", "PET":"Plastic",

    # Metal
    "Alubuechse":"Metal", "Kaffeekapsel":"Metal", "Aluminium":"Metal",

    # Glass
    "Glasflasche":"Glass", "Glass":"Glass",

    # Paper/Cardboard
    "Papier":"Paper_Cardboard", "Pappe":"Paper_Cardboard",
    "Paper":"Paper_Cardboard", "Cardboard":"Paper_Cardboard",

    # Explicit one-to-ones
    "Wagon":"Wagon",
    "Container":"Container",

    # Everything else not listed here falls to "Other"
}

def norm_label(raw: str) -> str:
    n = RAW_TO_NORM.get((raw or "").strip())
    return n if n in TARGET_CLASSES else (n or "Other")

def parse_ls(json_path: Path) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of tasks in Label Studio export (list).")
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--examples", type=int, default=0,
                    help="record up to N example image names per normalized class (no copying).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lists_dir = out_dir / "lists"; lists_dir.mkdir(parents=True, exist_ok=True)

    tasks = parse_ls(Path(args.export_json))

    # Normalized counters
    inst_per_norm = {c: 0 for c in TARGET_CLASSES}
    imgs_per_norm = {c: set() for c in TARGET_CLASSES}
    examples = {c: [] for c in TARGET_CLASSES}

    # Raw counters for transparency
    inst_per_raw = defaultdict(int)
    imgs_per_raw = defaultdict(set)
    raw_to_norm = {}

    for t in tasks:
        data = t.get("data", {})
        img_name = Path(data.get("image","")).name
        if not img_name:
            continue

        anns = t.get("annotations") or []
        present_norm = set()
        present_raw = set()

        for ann in anns:
            for res in (ann.get("result") or []):
                value = res.get("value", {}) or {}
                raw = ""
                for key in ("brushlabels","rectanglelabels","polygonlabels","labels"):
                    v = value.get(key)
                    if isinstance(v, list) and v:
                        raw = str(v[0]); break
                if not raw:
                    continue

                norm = norm_label(raw)
                if norm not in TARGET_CLASSES:
                    norm = "Other"

                # normalized tallies
                inst_per_norm[norm] += 1
                present_norm.add(norm)

                # raw tallies
                inst_per_raw[raw] += 1
                present_raw.add(raw)
                raw_to_norm[raw] = norm

        for nlab in present_norm:
            imgs_per_norm[nlab].add(img_name)
            if args.examples > 0 and len(examples[nlab]) < args.examples:
                examples[nlab].append(img_name)

        for rlab in present_raw:
            imgs_per_raw[rlab].add(img_name)

    # Write per-class image lists (normalized)
    for c in TARGET_CLASSES:
        with open(lists_dir / f"{c}.txt", "w", encoding="utf-8") as f:
            for nm in sorted(imgs_per_norm[c]):
                f.write(nm + "\n")

    # 8-row summary CSV
    rows = []
    for c in TARGET_CLASSES:
        imgs = sorted(imgs_per_norm[c])
        rows.append({
            "class": c,
            "instances": inst_per_norm[c],
            "images": len(imgs),
            "example_1": imgs[0] if imgs else ""
        })
    pd.DataFrame(rows, columns=["class","instances","images","example_1"]) \
      .to_csv(out_dir / "labels_summary_8.csv", index=False, encoding="utf-8-sig")

    # Raw -> normalized mapping with counts (transparency report)
    rows_map: List[Tuple[str,str,int,int]] = []
    for rlab, inst_cnt in sorted(inst_per_raw.items(), key=lambda x: (-x[1], x[0])):
        rows_map.append((
            rlab,
            raw_to_norm.get(rlab, "Other"),
            inst_cnt,
            len(imgs_per_raw[rlab])
        ))
    pd.DataFrame(rows_map, columns=["label_raw","label_norm","instances","images"]) \
      .to_csv(out_dir / "raw_to_norm_counts.csv", index=False, encoding="utf-8-sig")

    print(f"[done] 8-class CSV     : {out_dir/'labels_summary_8.csv'}")
    print(f"[done] Raw→Norm counts : {out_dir/'raw_to_norm_counts.csv'}")
    print(f"[done] Lists per class : {lists_dir}")

if __name__ == "__main__":
    main()
