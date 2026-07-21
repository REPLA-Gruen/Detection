# -*- coding: utf-8 -*-
"""
Label Studio -> quick label statistics + (optional) rendered examples.

Fixes:
- Auto-detects LS export shape: list | {"tasks":[...]} | {"data":...}
- Reads annotations from 'annotations' OR 'completions', 'result' inside each
- Finds image path in data['image'] or data['file_upload']
- Detailed debug logs about what it parsed
- Robust box/polygon/RLE handling



Run structure:
# Activate your (FMD) venv first
# conda activate FMD     # or:  .\.venv\Scripts\activate

python C:\Hadi\Foreign_Material\Code\Python\tools\1-ls_label_stats_v2.py `
  --export_json "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\project-2-at-2026-01-06-10-09-5c186fe2.json" `
  --images_dir  "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\images" `
  --out_dir     "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\stats" `
  --examples 30 `
  --render true `
  --box_thick 8 --font_scale 1.0 --font_thick 2

"""

import argparse, json, os
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------ RLE (Label Studio brush) ------------------
class _InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0
    def read(self, size):
        out = self.data[self.i: self.i + size]
        self.i += size
        return int(out, 2)

def _access_bit(data, num):
    base = int(num // 8); shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift

def _bytes2bit(data):
    return "".join([str(_access_bit(data, i)) for i in range(len(data) * 8)])

def rle_to_mask(rle, height, width):
    # rle is bytes array (Label Studio) -> to bits string
    rle_input = _InputStream(_bytes2bit(rle))
    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1
    img = np.reshape(out, [height, width, 4])[:, :, 3]
    return (img > 0).astype(np.uint8)
# --------------------------------------------------------------

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _rand_color(key: str):
    s = abs(hash(key)) % (256**3)
    return int((s>>16)&255), int((s>>8)&255), int(s&255)

def _norm_label(s: str) -> str:
    # simple normalization for grouping (keep accents/spaces minimal)
    return " ".join(str(s).strip().replace("_","-").split())

def _draw_labelled_box(img, box, color, label, font_scale=0.9, box_thick=6, font_thick=2):
    x,y,w,h = box
    cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=box_thick)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        y1 = max(0, y - th - 6)
        cv2.rectangle(img, (x, y1), (x + tw + 6, y), color, -1)
        cv2.putText(img, label, (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thick, cv2.LINE_AA)

def _mask_to_box(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]
    if not cnts: return None
    x,y,w,h = cv2.boundingRect(np.vstack(cnts))
    return (x,y,w,h)

def _rect_to_box(value, H, W):
    ow = int(value.get("original_width",  W) or W)
    oh = int(value.get("original_height", H) or H)
    x = float(value.get("x",0)); y=float(value.get("y",0))
    w = float(value.get("width",0)); h=float(value.get("height",0))
    if max(x,y,w,h) <= 100.0:   # percentages
        px = int(round(x*ow/100.0)); py = int(round(y*oh/100.0))
        pw = int(round(w*ow/100.0)); ph = int(round(h*oh/100.0))
    else:  # absolute in original -> map if resized
        px = int(round(x*W/ow)); py = int(round(y*H/oh))
        pw = int(round(w*W/ow)); ph = int(round(h*H/oh))
    px = max(0, min(px, W-1)); py = max(0, min(py, H-1))
    pw = max(1, min(pw, W-px)); ph = max(1, min(ph, H-py))
    return (px,py,pw,ph)

def _iter_tasks(root):
    """Yield (data_dict, annotations_list) for each task."""
    # Accept list, dict with 'tasks', or dict of tasks
    if isinstance(root, list):
        tasks = root
    elif isinstance(root, dict) and "tasks" in root and isinstance(root["tasks"], list):
        tasks = root["tasks"]
    else:
        # sometimes LS uses {"export_type":..., "items":[...]} or similar
        for k in ("items", "results", "records"):
            if isinstance(root, dict) and k in root and isinstance(root[k], list):
                tasks = root[k]; break
        else:
            tasks = [root] if isinstance(root, dict) else []

    for t in tasks:
        data = t.get("data", {}) if isinstance(t, dict) else {}
        anns = t.get("annotations") or t.get("completions") or []
        # Some exports wrap annotations under {"annotations":[{"result":[...]}, ...]}
        if isinstance(anns, dict) and "result" in anns:
            anns = [anns]
        yield data, anns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_json", required=True)
    ap.add_argument("--images_dir",  required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--examples", type=int, default=30)
    ap.add_argument("--render",   type=str, default="true")
    ap.add_argument("--box_thick", type=int, default=6)
    ap.add_argument("--font_scale", type=float, default=0.9)
    ap.add_argument("--font_thick", type=int, default=2)
    args = ap.parse_args()

    export_json = Path(args.export_json)
    images_dir  = Path(args.images_dir)
    out_dir     = Path(args.out_dir)
    do_render   = str(args.render).lower() in ("1","true","yes","y")
    n_examples  = int(args.examples)

    _ensure_dir(out_dir)
    lists_dir   = out_dir / "lists";            _ensure_dir(lists_dir)
    ex_dir      = out_dir / "examples";         _ensure_dir(ex_dir)
    ex_r_dir    = out_dir / "examples_render";  _ensure_dir(ex_r_dir)

    # ---------------- load JSON ----------------
    with open(export_json, "r", encoding="utf-8") as f:
        root = json.load(f)

    # quick debug dump
    dbg = out_dir / "debug_detect.json"
    with open(dbg, "w", encoding="utf-8") as f:
        if isinstance(root, list) and root:
            json.dump({"first_task_keys": list(root[0].keys())}, f, ensure_ascii=False, indent=2)
        elif isinstance(root, dict):
            json.dump({"root_keys": list(root.keys())}, f, ensure_ascii=False, indent=2)

    # ---------------- parse ----------------
    # stats
    inst_per_label = Counter()
    img_per_label  = Counter()
    imgs_with_label = defaultdict(set)

    # for example saving
    examples_for = defaultdict(list)

    # scan pass
    n_tasks = 0
    with tqdm(desc="scan", unit="task") as pbar:
        for data, ann_list in _iter_tasks(root):
            n_tasks += 1
            # image path may be in 'image' or 'file_upload'
            rel = data.get("image") or data.get("file_upload") or ""
            if not rel:
                pbar.update(1); continue
            img_path = images_dir / Path(rel).name

            # iterate annotations
            have_any = False
            for ann in (ann_list or []):
                results = ann.get("result") or []
                if not isinstance(results, list): continue
                for r in results:
                    value = r.get("value", {}) or {}
                    # label field
                    label = None
                    for key in ("rectanglelabels","polygonlabels","brushlabels","labels"):
                        v = value.get(key)
                        if isinstance(v, list) and v: 
                            label = str(v[0]); break
                    if not label: 
                        continue
                    lbl_raw = label
                    lbl_nor = _norm_label(label)
                    inst_per_label[lbl_raw] += 1
                    imgs_with_label[lbl_raw].add(img_path.name)
                    have_any = True
            pbar.update(1)

    # finalize image counts
    for lbl, s in imgs_with_label.items():
        img_per_label[lbl] = len(s)

    # ---------------- save CSV + lists ----------------
    labels_sorted = sorted(inst_per_label.keys(), key=lambda k: (-inst_per_label[k], k.lower()))
    rows = []
    for lbl in labels_sorted:
        rows.append({
            "label_raw": lbl,
            "label_nor": _norm_label(lbl),
            "instances": inst_per_label[lbl],
            "images": img_per_label[lbl],
            "example_1": next(iter(imgs_with_label[lbl])) if imgs_with_label[lbl] else "",
        })
        # list file
        with open(lists_dir / f"{lbl}.txt", "w", encoding="utf-8") as f:
            for nm in sorted(imgs_with_label[lbl]):
                f.write(nm + "\n")

    csv_path = out_dir / "labels_summary.csv"
    pd.DataFrame(rows, columns=["label_raw","label_nor","instances","images","example_1"]).to_csv(csv_path, index=False)

    # ---------------- collect examples (copy + render) ----------------
    if do_render and n_examples > 0 and labels_sorted:
        # Build a quick index of first N per class
        for lbl in labels_sorted:
            for nm in list(imgs_with_label[lbl])[:n_examples]:
                if nm not in examples_for[lbl]:
                    examples_for[lbl].append(nm)

        # We need boxes/masks to render -> re-parse, but now only if needed
        # To avoid quadratic cost, build a map image->all (label,shape)
        wants = {nm for lst in examples_for.values() for nm in lst}
        shapes_by_image = defaultdict(list)

        # second pass (parse shapes)
        for data, ann_list in _iter_tasks(root):
            rel = data.get("image") or data.get("file_upload") or ""
            if not rel: continue
            nm = Path(rel).name
            if nm not in wants: continue

            # Try to read image for size
            img_path = images_dir / nm
            img = cv2.imread(str(img_path))
            if img is None: continue
            H, W = img.shape[:2]

            for ann in (ann_list or []):
                results = ann.get("result") or []
                for r in results:
                    value = r.get("value", {}) or {}
                    # label
                    label = None
                    for key in ("rectanglelabels","polygonlabels","brushlabels","labels"):
                        v = value.get(key)
                        if isinstance(v, list) and v:
                            label = str(v[0]); break
                    if not label: 
                        continue
                    color = _rand_color(label)

                    # detect shape type
                    if value.get("format") == "rle" and "rle" in value:
                        try:
                            oh = int(r.get("original_height", H) or H)
                            ow = int(r.get("original_width",  W) or W)
                            mask = rle_to_mask(value["rle"], oh, ow)
                            if (oh,ow)!=(H,W):
                                mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
                            box = _mask_to_box((mask*255).astype(np.uint8))
                            if box: shapes_by_image[nm].append(("box", label, color, box))
                        except Exception:
                            pass
                    elif "points" in value and isinstance(value["points"], list):
                        # polygon -> mask -> box
                        pts = np.array(value["points"], dtype=np.float32)
                        if pts.max() <= 1.0:
                            pts[:,0] *= W; pts[:,1] *= H
                        poly = np.round(pts).astype(np.int32)
                        m = np.zeros((H,W), dtype=np.uint8)
                        cv2.fillPoly(m, [poly], 255)
                        box = _mask_to_box(m)
                        if box: shapes_by_image[nm].append(("box", label, color, box))
                    elif {"x","y","width","height"}.issubset(value.keys()):
                        box = _rect_to_box(value, H, W)
                        shapes_by_image[nm].append(("box", label, color, box))

        # write rendered images
        for lbl, nmlist in examples_for.items():
            for nm in nmlist:
                src = images_dir / nm
                img = cv2.imread(str(src))
                if img is None: continue
                # draw
                for tp, label, color, box in shapes_by_image.get(nm, []):
                    if tp == "box":
                        _draw_labelled_box(img, box, color, label,
                                           font_scale=float(args.font_scale),
                                           box_thick=int(args.box_thick),
                                           font_thick=int(args.font_thick))
                # save under per-class folder for convenience
                _ensure_dir(ex_dir / lbl); _ensure_dir(ex_r_dir / lbl)
                cv2.imwrite(str(ex_dir / lbl / nm), img)  # non-rendered copy not needed? keep identical for simplicity
                cv2.imwrite(str(ex_r_dir / lbl / nm), img)

    print(f"[done] CSV   : {csv_path}")
    print(f"[done] Lists : {lists_dir}")
    print(f"[done] Examples (raw)   : {ex_dir}")
    print(f"[done] Examples (render): {ex_r_dir}")
    print("[note] If the CSV is only headers, annotations were not found. Check out_dir/debug_detect.json to see the detected structure.")

if __name__ == "__main__":
    main()
