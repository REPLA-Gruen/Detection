# cm_and_render_yolov8_no_bg.py
# Creates per-split (train/val/test) 8x8 confusion matrices WITHOUT "background",
# plus per-class FN/FP tables, and rendered images.
#
# Usage (PowerShell example below the code)

import os, sys, json, math, shutil
from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:
    print("[fatal] ultralytics missing: pip install ultralytics", file=sys.stderr); raise

# ---------------- Utils ----------------

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def yolo_txt_to_boxes(txt_path, img_w, img_h):
    """Read YOLO label file -> list of (cls, x1,y1,x2,y2)."""
    boxes = []
    if not Path(txt_path).exists():
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 and len(parts) != 6:
                # allow optional conf col; ignore if present
                if len(parts) < 5:
                    continue
            cls = int(float(parts[0]))
            cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            x1 = (cx - w/2.0) * img_w
            y1 = (cy - h/2.0) * img_h
            x2 = (cx + w/2.0) * img_w
            y2 = (cy + h/2.0) * img_h
            # clip
            x1 = max(0, min(img_w-1, x1)); y1 = max(0, min(img_h-1, y1))
            x2 = max(0, min(img_w-1, x2)); y2 = max(0, min(img_h-1, y2))
            if x2 > x1 and y2 > y1:
                boxes.append((cls, x1, y1, x2, y2))
    return boxes

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    return inter / max(1e-9, (area_a + area_b - inter))

def greedy_match(gts, preds, iou_thr):
    """
    gts:  list of (gt_cls, x1,y1,x2,y2)
    preds:list of (pr_cls, x1,y1,x2,y2,conf)
    Returns:
      matches: list of (gt_idx, pr_idx)
      gt_unmatched_idx: list
      pr_unmatched_idx: list
    Match by maximum IoU per GT (greedy).
    """
    if not gts or not preds:
        return [], list(range(len(gts))), list(range(len(preds)))
    iou_mat = np.zeros((len(gts), len(preds)), dtype=np.float32)
    for i, (_, gx1,gy1,gx2,gy2) in enumerate(gts):
        for j, (_, px1,py1,px2,py2,_) in enumerate(preds):
            iou_mat[i,j] = iou_xyxy((gx1,gy1,gx2,gy2), (px1,py1,px2,py2))
    matches = []
    used_g = set(); used_p = set()
    # Greedy: repeatedly pick best IoU left above threshold
    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        best = iou_mat[i,j]
        if best < iou_thr: break
        if i not in used_g and j not in used_p:
            matches.append((i,j))
            used_g.add(i); used_p.add(j)
        iou_mat[i,:] = -1.0; iou_mat[:,j] = -1.0
    gt_un = [i for i in range(len(gts)) if i not in used_g]
    pr_un = [j for j in range(len(preds)) if j not in used_p]
    return matches, gt_un, pr_un

# consistent colors per class
def class_color(idx):
    # fixed palette
    palette = [
        (50,205,50),   # 0 BAW-Sack
        (0,165,255),   # 1 Plastic
        (255,0,0),     # 2 Metal
        (255,255,0),   # 3 Glass
        (255,140,0),   # 4 Paper/Cardboard
        (128,0,128),   # 5 Other
        (0,255,255),   # 6 Wagon
        (0,0,255),     # 7 Container
    ]
    return palette[idx % len(palette)]

def put_label(img, text, x, y, bg_color, font_scale, font_thick):
    tsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    tw, th = tsize
    y = max(th+4, y)
    cv2.rectangle(img, (x, y-th-4), (x+tw+6, y+2), bg_color, -1)
    cv2.putText(img, text, (x+3, y-2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thick, cv2.LINE_AA)

def draw_box(img, x1,y1,x2,y2, color, thick):
    cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, thick)

# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best.pt")
    ap.add_argument("--data_yaml", required=True, help="Path to dataset_norm8.yaml")
    ap.add_argument("--out_dir", required=True, help="Where to save CMs, CSVs, renders")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.45, help="NMS IoU for predictor")
    ap.add_argument("--match_iou", type=float, default=0.5, help="IoU threshold for CM matching")
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--render", type=str, default="true")
    ap.add_argument("--box_thick", type=int, default=8)
    ap.add_argument("--font_scale", type=float, default=1.4)
    ap.add_argument("--font_thick", type=int, default=3)
    args = ap.parse_args()

    render = str(args.render).lower() in ("1","true","yes","y")
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.data_yaml)
    names = cfg.get("names")
    if isinstance(names, dict):
        # convert dict 0..7 -> list
        names = [names[i] for i in range(len(names))]
    assert isinstance(names, (list,tuple)) and len(names) == 8, "Expect exactly 8 class names."

    # Resolve splits from yaml root
    base = cfg.get("path", None)
    if base is None:
        base = Path(args.data_yaml).parent
    base = Path(base)

    def split_dir(key):
        p = cfg.get(key)  # like images\train
        return (base / p).resolve()

    splits = {
        "train": split_dir("train"),
        "val":   split_dir("val"),
        "test":  split_dir("test"),
    }

    # Load model
    model = YOLO(args.weights)
    # Predictor settings
    predictor_kwargs = dict(conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, max_det=args.max_det, verbose=False)

    # For each split
    for split_name, img_dir in splits.items():
        if not img_dir.exists():
            print(f"[warn] split '{split_name}' missing at: {img_dir}")
            continue

        out_split = out_root / split_name
        out_split.mkdir(parents=True, exist_ok=True)
        out_render = out_split / "renders"
        if render: out_render.mkdir(parents=True, exist_ok=True)

        # Confusion matrix (8x8), FN and FP counters
        K = len(names)
        cm = np.zeros((K, K), dtype=np.int64)  # [true, pred]
        fn = np.zeros((K,), dtype=np.int64)
        fp = np.zeros((K,), dtype=np.int64)

        # Walk images
        image_paths = [p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
        if not image_paths:
            print(f"[warn] No images in {img_dir}")
            continue

        pbar = tqdm(image_paths, desc=f"{split_name}: infer+CM")
        for imp in pbar:
            im = cv2.imread(str(imp))
            if im is None: 
                continue
            H, W = im.shape[:2]
            lbl = (imp.parent.parent / "labels" / imp.parent.name / (imp.stem + ".txt")).resolve()
            gts = yolo_txt_to_boxes(lbl, W, H)  # (cls,x1,y1,x2,y2)

            # predict
            res = model.predict(source=str(imp), **predictor_kwargs)[0]
            preds = []
            if res and res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for (x1,y1,x2,y2), c, s in zip(xyxy, cls, conf):
                    # clip to image
                    x1 = max(0, min(W-1, float(x1))); y1 = max(0, min(H-1, float(y1)))
                    x2 = max(0, min(W-1, float(x2))); y2 = max(0, min(H-1, float(y2)))
                    if x2 > x1 and y2 > y1:
                        preds.append((int(c), x1,y1,x2,y2, float(s)))

            # match
            matches, gt_un, pr_un = greedy_match(gts, preds, iou_thr=args.match_iou)

            # fill confusion matrix WITHOUT background: only matched pairs land in CM
            for gi, pj in matches:
                t_cls = gts[gi][0]
                p_cls = preds[pj][0]
                if 0 <= t_cls < K and 0 <= p_cls < K:
                    cm[t_cls, p_cls] += 1

            # Count FN per true class and FP per predicted class (separate tables)
            for gi in gt_un:
                t_cls = gts[gi][0]
                if 0 <= t_cls < K: fn[t_cls] += 1
            for pj in pr_un:
                p_cls = preds[pj][0]
                if 0 <= p_cls < K: fp[p_cls] += 1

            # render
            if render:
                vis = im.copy()
                # draw GT boxes: thin, green-ish with class label
                for (t_cls, x1,y1,x2,y2) in gts:
                    color = (0,255,0)
                    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), color, max(1, args.box_thick//2))
                    put_label(vis, f"GT:{names[t_cls]}", int(x1), int(y1)-2, color, args.font_scale*0.9, args.font_thick)

                # draw predictions: thicker, class-color, label with conf
                for (p_cls, x1,y1,x2,y2, conf) in preds:
                    color = class_color(p_cls)
                    draw_box(vis, x1,y1,x2,y2, color, args.box_thick)
                    put_label(vis, f"{names[p_cls]} {conf:.2f}", int(x1), int(y1)-2, color, args.font_scale, args.font_thick)

                cv2.imwrite(str(out_render / imp.name), vis)

        # Save confusion matrix & FN/FP
        df_cm = pd.DataFrame(cm, index=[f"T:{n}" for n in names], columns=[f"P:{n}" for n in names])
        df_cm.to_csv(out_split / "confusion_no_bg.csv", index=True)

        # Normalize by true count (row-wise) for readability
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = (cm / row_sums).round(4)
        pd.DataFrame(cm_norm, index=[f"T:{n}" for n in names], columns=[f"P:{n}" for n in names]).to_csv(out_split / "confusion_no_bg_normalized.csv", index=True)

        # Save FN / FP tables
        pd.DataFrame({"class":names, "FN":fn}).to_csv(out_split / "per_class_FN.csv", index=False)
        pd.DataFrame({"class":names, "FP":fp}).to_csv(out_split / "per_class_FP.csv", index=False)

        # Quick PNG heatmap (no seaborn, pure cv2)
        try:
            # scale to 0..255
            cm_img = (cm_norm * 255).astype(np.uint8)
            # upscale for visibility
            scale = 32
            cm_big = cv2.resize(cm_img, (cm_img.shape[1]*scale, cm_img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
            cm_big = cv2.applyColorMap(cm_big, cv2.COLORMAP_VIRIDIS)
            # write axis labels
            step = scale
            # class labels on Y (true)
            for r, name in enumerate(names):
                cv2.putText(cm_big, name[:12], (3, (r*step)+int(step*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            # class labels on X (pred)
            for c, name in enumerate(names):
                cv2.putText(cm_big, name[:12], ((c*step)+4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_split / "confusion_no_bg.png"), cm_big)
        except Exception:
            pass

        print(f"[done] {split_name}: CM (no-bg) and renders saved to: {out_split}")

if __name__ == "__main__":
    main()
