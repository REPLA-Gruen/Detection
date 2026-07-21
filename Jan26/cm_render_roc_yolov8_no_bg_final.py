#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
YOLOv8 evaluation utilities for 8 FO classes (no background row/col in CM)
- Renders predictions
- Builds per-split 8x8 confusion matrices (correct IoU matching)
- Saves CSVs and heatmaps
- Computes per-class ROC curves + AUC (IoU-based matching)

Usage (PowerShell, same style you use):
python C:\Hadi\Foreign_Material\Code\Python\tools\cm_render_roc_yolov8_no_bg_final.py `
  --weights "C:\Hadi\Foreign_Material\Runs\yolov8m_norm8_960\exp\weights\best.pt" `
  --data_yaml "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\YOLO_norm8\dataset_norm8.yaml" `
  --out_dir  "C:\Hadi\Foreign_Material\Runs\yolov8m_norm8_960\confusion_no_bg" `
  --imgsz 960 --device 0 `
  --conf 0.25 --iou 0.45 --match_iou 0.50 `
  --render true --box_thick 8 --font_scale 1.5 --font_thick 3
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.ops import xywh2xyxy

# --------------------------
# Helpers
# --------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=str)
    p.add_argument("--data_yaml", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--imgsz", default=960, type=int)
    p.add_argument("--device", default="0", type=str)  # "0" | "cpu" | "cuda:0"
    p.add_argument("--conf", default=0.25, type=float)      # score threshold for rendering and CM
    p.add_argument("--iou", default=0.45, type=float)       # NMS IoU (YOLO inference)
    p.add_argument("--match_iou", default=0.50, type=float) # IoU for matching preds↔labels (CM/ROC)
    p.add_argument("--render", default="true", type=str)    # "true"/"false"
    p.add_argument("--box_thick", default=8, type=int)
    p.add_argument("--font_scale", default=1.2, type=float)
    p.add_argument("--font_thick", default=3, type=int)
    return p.parse_args()

def to_bool(x: str) -> bool:
    return str(x).lower() in {"1","true","yes","y","t"}

def device_string(d: str) -> str:
    # Accept "0" -> "cuda:0", "cpu" -> "cpu", "cuda:1" -> "cuda:1"
    d = str(d).strip().lower()
    if d == "cpu": return "cpu"
    if d.startswith("cuda"): return d
    # bare index
    if d.isdigit():
        return f"cuda:{d}"
    return d

def load_yaml(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_label_files(labels_dir: Path):
    return sorted([p for p in Path(labels_dir).glob("*.txt")])

def label_file_to_image_path(label_path: Path, images_dir: Path):
    stem = label_path.stem
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

def read_yolo_txt(txt_path: Path):
    # returns np.array [N,5] -> (cls, cx, cy, w, h) normalized
    if not txt_path.exists():
        return np.zeros((0,5), dtype=np.float32)
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: 
                continue
            rows.append([float(x) for x in parts])
    if not rows:
        return np.zeros((0,5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def draw_box(img, xyxy, color, label, box_thick=3, font_scale=0.7, font_thick=2, inside=False):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, box_thick)
    if label:
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        if inside:
            tx = max(x1+3, 0); ty = max(y1+th+3, th+3)
            cv2.rectangle(img, (tx-2, ty-th-base-2), (tx+tw+2, ty+2), (0,0,0), -1)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thick, cv2.LINE_AA)
        else:
            bx1, by1 = x1, max(0, y1 - th - 6)
            cv2.rectangle(img, (bx1, by1), (bx1+tw+6, by1+th+6), color, -1)
            cv2.putText(img, label, (bx1+3, by1+th+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thick, cv2.LINE_AA)

def cm_save_csv(matrix8: np.ndarray, names: list, out_csv: Path, per_class_csv: Path, fp_csv: Path):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [int(v) for v in matrix8[i].tolist()])

    tp = np.diag(matrix8).astype(int)
    fn = matrix8.sum(axis=1).astype(int) - tp
    fp = matrix8.sum(axis=0).astype(int) - tp

    with open(per_class_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class","TP","FP","FN","support"])
        for i, n in enumerate(names):
            w.writerow([n, int(tp[i]), int(fp[i]), int(fn[i]), int(matrix8[i].sum())])

    with open(fp_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class","FP"])
        for i, n in enumerate(names):
            w.writerow([n, int(fp[i])])

def plot_cm(matrix8: np.ndarray, names: list, out_png: Path):
    plt.figure(figsize=(7,6))
    norm = matrix8 / np.maximum(matrix8.sum(axis=1, keepdims=True), 1)
    im = plt.imshow(norm, interpolation="nearest", cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.title("Confusion Matrix (row-normalized)")
    for i in range(matrix8.shape[0]):
        for j in range(matrix8.shape[1]):
            val = norm[i, j]
            txt = f"{val*100:.1f}%\n({int(matrix8[i,j])})"
            c = "white" if val > 0.5 else "w"
            plt.text(j, i, txt, ha="center", va="center", color="white", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def trapezoid_auc(x, y):
    # x should be ascending
    return np.trapz(y, x)

# --------------------------
# Evaluation core
# --------------------------
def eval_split(
    model, split_name: str, images_dir: Path, labels_dir: Path, names: list, args, out_root: Path
):
    """
    Runs per-image inference, renders (optionally), builds CM, exports CSV/PNG,
    and computes per-class ROC/AUC.
    """
    NC = len(names)
    out_split = out_root / split_name
    out_vis = out_split / "renders"
    ensure_dir(out_split); ensure_dir(out_vis)

    # ConfusionMatrix from Ultralytics expects:
    # det:  [N,6] -> x1,y1,x2,y2,conf,cls
    # labels:[M,5] -> cls,x1,y1,x2,y2
    cm = ConfusionMatrix(nc=NC, conf=args.conf, iou_thres=args.match_iou)

    # For ROC sweep we need to re-run matching at many thresholds.
    # We'll cache all per-image detections and labels.
    cache_dets = {}   # img_id -> ndarray [Nd,6]
    cache_gts  = {}   # img_id -> ndarray [Ng,5] (cls, x1,y1,x2,y2)

    label_files = list_label_files(labels_dir)
    color_table = [
        (60,180,75),   # BAW-Sack
        (255,200,87),  # Plastic
        (255,111,105), # Metal
        (67,170,139),  # Glass
        (87,117,144),  # Paper_Cardboard
        (199,0,57),    # Other
        (40,40,210),   # Wagon (blue-ish for readability)
        (230,90,0),    # Container (orange)
    ]
    wagon_idx = names.index("Wagon") if "Wagon" in names else 6
    cont_idx  = names.index("Container") if "Container" in names else 7

    for txt in tqdm(label_files, desc=f"[{split_name}] evaluating"):
        img_path = label_file_to_image_path(txt, images_dir)
        if img_path is None:
            continue

        gt_norm = read_yolo_txt(txt)  # [M,5] cls cx cy w h in 0..1
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        H, W = im.shape[:2]

        # convert GT to xyxy pixels (cls,x1,y1,x2,y2)
        if len(gt_norm):
            gt_xyxy = xywh2xyxy(gt_norm[:,1:5])
            gain = np.array([W, H, W, H], dtype=np.float32)
            gt_xyxy = (gt_xyxy * gain).astype(np.float32)
            labels_xyxy = np.concatenate(
                [gt_norm[:,[0]].astype(np.float32), gt_xyxy], axis=1
            ).astype(np.float32)
        else:
            labels_xyxy = np.zeros((0,5), dtype=np.float32)

        # model inference for this image
        res = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            device=device_string(args.device),
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )[0]

        # predictions in xyxy pixels
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes
            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            conf = b.conf.cpu().numpy().astype(np.float32).reshape(-1,1)
            cls  = b.cls.cpu().numpy().astype(np.float32).reshape(-1,1)
            dets = np.concatenate([xyxy, conf, cls], axis=1)
        else:
            dets = np.zeros((0,6), dtype=np.float32)

        # cache for ROC
        cache_dets[img_path] = dets
        cache_gts[img_path]  = labels_xyxy

        # update confusion matrix at current args.conf (matches inside)
        cm.process_batch(dets, labels_xyxy)

        # rendering
        if to_bool(args.render):
            vis = im.copy()
            # draw ground-truth boxes thin (optional)
            # draw predictions with class colors
            for d in dets:
                x1,y1,x2,y2,cf,cl = d
                cls_i = int(cl)
                color = color_table[cls_i % len(color_table)]
                label = f"{names[cls_i]} {cf:.2f}"
                # put Wagon/Container text *inside* box so it remains readable
                inside = cls_i in (wagon_idx, cont_idx)
                draw_box(vis, (x1,y1,x2,y2), color, label,
                         box_thick=args.box_thick, font_scale=args.font_scale,
                         font_thick=args.font_thick, inside=inside)
            out_img = out_vis / (Path(img_path).stem + ".jpg")
            cv2.imwrite(str(out_img), vis)

    # CM save (drop background row/col)
    M = cm.matrix.astype(int)        # (NC+1, NC+1)
    M8 = M[:NC, :NC]                 # 8x8, no background
    cm_save_csv(
        matrix8=M8,
        names=names,
        out_csv=out_split / "confusion_no_bg.csv",
        per_class_csv=out_split / "per_class_counts.csv",
        fp_csv=out_split / "per_class_FP.csv"
    )
    plot_cm(M8, names, out_split / "confusion_no_bg.png")

    # --------------------------
    # ROC / AUC per class
    # --------------------------
    # We sweep thresholds and recompute TP/FP/FN counts with same IoU policy.
    thresholds = np.linspace(0.001, 0.999, 50)  # 50 points
    roc_data = {c: {"tpr": [], "fpr": [], "thr": []} for c in range(NC)}

    def counts_at_threshold(thr):
        # compute TP/FP/FN per class at threshold thr
        cm_thr = ConfusionMatrix(nc=NC, conf=thr, iou_thres=args.match_iou)
        for img_path, dets in cache_dets.items():
            # filter dets by thr
            if len(dets):
                keep = dets[:,4] >= thr
                det_use = dets[keep].astype(np.float32)
            else:
                det_use = np.zeros((0,6), dtype=np.float32)
            labels_xyxy = cache_gts[img_path]
            cm_thr.process_batch(det_use, labels_xyxy)
        Mthr = cm_thr.matrix.astype(int)[:NC,:NC]
        tp = np.diag(Mthr).astype(int)
        fn = Mthr.sum(axis=1).astype(int) - tp
        fp = Mthr.sum(axis=0).astype(int) - tp
        tn = Mthr.sum() - (tp + fp + fn)
        return tp, fp, fn, tn

    for thr in tqdm(thresholds, desc=f"[{split_name}] ROC sweep"):
        tp, fp, fn, tn = counts_at_threshold(float(thr))
        # TPR = TP / (TP+FN), FPR = FP / (FP+TN)
        denom_pos = (tp + fn).astype(float)
        denom_neg = (fp + tn).astype(float)
        tpr = np.divide(tp, np.maximum(denom_pos, 1e-9))
        fpr = np.divide(fp, np.maximum(denom_neg, 1e-9))
        for c in range(NC):
            roc_data[c]["tpr"].append(float(tpr[c]))
            roc_data[c]["fpr"].append(float(fpr[c]))
            roc_data[c]["thr"].append(float(thr))

    # Save ROC CSVs and a combined PNG
    roc_dir = out_split / "roc"
    ensure_dir(roc_dir)
    plt.figure(figsize=(7,6))
    for c in range(NC):
        fpr = np.array(roc_data[c]["fpr"])
        tpr = np.array(roc_data[c]["tpr"])
        # sort by FPR ascending for AUC
        order = np.argsort(fpr)
        fpr = fpr[order]; tpr = tpr[order]
        auc = trapezoid_auc(fpr, tpr)
        # CSV
        with open(roc_dir / f"roc_{names[c]}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["threshold","FPR","TPR"])
            for i in range(len(fpr)):
                w.writerow([roc_data[c]["thr"][order[i]], fpr[i], tpr[i]])
            w.writerow(["AUC", auc, ""])
        # plot
        plt.plot(fpr, tpr, label=f"{names[c]} (AUC={auc:.3f})")
    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-class ROC (IoU-based matching)")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_dir / "roc_all_classes.png", dpi=200)
    plt.close()

    # Return raw matrix for caller summary if needed
    return M8

# --------------------------
# Main
# --------------------------
def main():
    args = parse_args()
    args.render = to_bool(args.render)

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    data = load_yaml(args.data_yaml)
    # expected keys: train, val, test, and names (0..7)
    names_map = data.get("names", None)
    if isinstance(names_map, dict):
        names = [names_map[i] for i in sorted(names_map.keys())]
    else:
        names = list(names_map) if names_map else [
            "BAW-Sack","Plastic","Metal","Glass","Paper_Cardboard","Other","Wagon","Container"
        ]
    assert len(names) == 8, f"Expecting 8 classes, got {len(names)}"

    # Resolve dirs
    def to_dir(path_or_file):
        p = Path(path_or_file)
        # Ultralytics allows list files too; we assume directories with images/labels
        return p

    # dataset_norm8.yaml typical structure:
    # path/train/images, path/train/labels etc OR absolute split dirs
    # Try to infer split roots
    def split_dirs(split_key):
        # If YAML has 'train': 'PATH/images', derive labels as replacing '/images' with '/labels'
        sp = Path(data[split_key])
        if sp.is_dir():
            # sp is images dir; labels beside
            images = sp
            labels = sp.parent / "labels"
        else:
            raise RuntimeError(f"Unsupported YAML split path: {sp}")
        return images, labels

    train_img, train_lbl = split_dirs("train")
    val_img,   val_lbl   = split_dirs("val")
    test_img,  test_lbl  = split_dirs("test")

    # Load model
    model = YOLO(args.weights)
    # device handling
    dev = device_string(args.device)
    model.to(dev)

    # Evaluate splits
    M_train = eval_split(model, "train", Path(train_img), Path(train_lbl), names, args, out_root)
    M_val   = eval_split(model, "val",   Path(val_img),   Path(val_lbl),   names, args, out_root)
    M_test  = eval_split(model, "test",  Path(test_img),  Path(test_lbl),  names, args, out_root)

    # Save a tiny JSON summary
    summary = {
        "names": names,
        "match_iou": args.match_iou,
        "conf_threshold": args.conf,
        "train_sum": int(M_train.sum()),
        "val_sum": int(M_val.sum()),
        "test_sum": int(M_test.sum()),
    }
    with open(Path(args.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()