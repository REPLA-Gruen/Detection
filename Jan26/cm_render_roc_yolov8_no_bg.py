#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# ---------- helpers

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def draw_label(img, text, xy, font_scale=0.9, thick=2, color=(255,255,255), outline=(0,0,0)):
    x, y = int(xy[0]), int(xy[1])
    ((w, h), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    cv2.rectangle(img, (x-4, y-h-6), (x + w + 4, y + 6), outline, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

def render_boxes(img, boxes_xyxy, cls_ids, scores, names, box_thick=3, font_scale=0.9, font_thick=2):
    for (x1,y1,x2,y2), cid, sc in zip(boxes_xyxy, cls_ids, scores):
        color = (0, 170, 255)  # amber-ish
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thick)
        # write class text inside the box for full-frame classes too
        tx = names.get(int(cid), str(cid))
        draw_label(img, f"{tx} {sc:.2f}", (max(int(x1)+6, 4), max(int(y1)+18, 18)), font_scale, font_thick)
    return img

def xywhn_to_xyxy(box, w, h):
    # box: (x_c, y_c, w, h) normalized
    x_c, y_c, bw, bh = box
    x1 = (x_c - bw/2) * w
    y1 = (y_c - bh/2) * h
    x2 = (x_c + bw/2) * w
    y2 = (y_c + bh/2) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def compute_matches(pred_xyxy, pred_cls, pred_scores, gt_xyxy, gt_cls, iou_thr=0.50):
    """
    Greedy match by IoU, return index pairs (pi->gi), and FP/FN sets.
    """
    if len(pred_xyxy)==0 and len(gt_xyxy)==0:
        return [], np.array([], int), np.array([], int)

    matches = []
    used_gt = set()
    # compute IoU matrix
    ious = box_iou(
        torch.tensor(pred_xyxy, dtype=torch.float32),
        torch.tensor(gt_xyxy,   dtype=torch.float32)
    ).numpy() if (len(pred_xyxy) and len(gt_xyxy)) else np.zeros((len(pred_xyxy), len(gt_xyxy)), dtype=np.float32)

    # For each prediction in descending score order, pick best IoU GT if class matches
    order = np.argsort(-pred_scores)
    for pi in order:
        if len(gt_xyxy)==0:
            break
        gi_cands = np.where(ious[pi] >= iou_thr)[0]
        if gi_cands.size == 0:
            continue
        # among candidates, require class match and max IoU
        gi_cands = [gi for gi in gi_cands if int(gt_cls[gi]) == int(pred_cls[pi]) and gi not in used_gt]
        if not gi_cands:
            continue
        gi = max(gi_cands, key=lambda g: ious[pi, g])
        matches.append((pi, gi))
        used_gt.add(gi)

    matched_pred = set(pi for pi,_ in matches)
    matched_gt   = set(gi for _,gi in matches)

    fp = np.array([i for i in range(len(pred_xyxy)) if i not in matched_pred], dtype=int)
    fn = np.array([i for i in range(len(gt_xyxy))   if i not in matched_gt],   dtype=int)
    return matches, fp, fn

# torch only where needed
import torch

def soft_to_one_vs_rest(labels, num_classes):
    y = np.zeros((labels.size, num_classes), dtype=np.int32)
    y[np.arange(labels.size), labels.astype(int)] = 1
    return y

def auc_from_roc(fpr, tpr):
    # trapezoidal rule
    return np.trapz(tpr, fpr)

# ---------- main

def evaluate_split(model, data_yaml, split, out_dir, class_names, imgsz=960, device=0, conf=0.25, iou_nms=0.45, match_iou=0.50,
                   render=False, box_thick=6, font_scale=1.0, font_thick=2):

    out_dir = ensure_dir(Path(out_dir) / split)
    # Force YOLOv8 to run on a specific split
    res = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, conf=conf, iou=iou_nms, save_txt=False, save=False, plots=False, verbose=False)
    # The dataloader used by Ultralytics is available via res.dataloader with same ordering as res.boxes

    # Collect ground truth from dataloader targets (YOLO format)
    # Each batch in res.dataloader gives (images, labels, paths,...)
    # We’ll fetch labels again to ensure alignment with image paths.

    # Build CM and ROC accumulators
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # row=GT, col=Pred

    # ROC stores per-class scores and labels (one-vs-rest)
    roc_scores = [[] for _ in range(num_classes)]
    roc_labels = [[] for _ in range(num_classes)]

    per_class_tp = np.zeros(num_classes, dtype=np.int64)
    per_class_fp = np.zeros(num_classes, dtype=np.int64)
    per_class_fn = np.zeros(num_classes, dtype=np.int64)
    support      = np.zeros(num_classes, dtype=np.int64)

    # Iterate over images by reading res.results which aligns with dataset ordering
    for ri, R in enumerate(tqdm(res.results, desc=f"[{split}] accumulating")):
        path = R.path
        imh, imw = R.orig_shape
        # GT boxes (xywh normalized, cls) from R.boxes.gt? Not provided. Use dataset labels:
        # res.files[ri] is image path; the dataset has labels at associated txt; Ultralytics loaded them.
        # Conveniently, R.boxes has .cls for predictions; ground truth is attached as R.boxes.gt if enabled in 8.3+, else read from ds.
        # Fallback: derive gt from R.boxes.orig_shape and R.boxes.data? Safer to open label file beside image.

        # Recover GT by reading label txt next to the image file (dataset already in YOLO format).
        lbl_path = Path(path).with_suffix(".txt").as_posix().replace("/images/", "/labels/")
        lbl_path = Path(lbl_path)
        gt_xyxy, gt_cls = [], []
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(float(parts[0]))
                        x,y,w,h = map(float, parts[1:5])
                        gt_xyxy.append(xywhn_to_xyxy((x,y,w,h), imw, imh))
                        gt_cls.append(c)
        gt_xyxy = np.stack(gt_xyxy, axis=0) if len(gt_xyxy) else np.zeros((0,4), dtype=np.float32)
        gt_cls  = np.array(gt_cls, dtype=np.int32) if len(gt_cls) else np.zeros((0,), dtype=np.int32)

        # Predictions
        if R.boxes is None or len(R.boxes)==0:
            pred_xyxy = np.zeros((0,4), dtype=np.float32)
            pred_cls  = np.zeros((0,), dtype=np.int32)
            pred_sc   = np.zeros((0,), dtype=np.float32)
        else:
            bx = R.boxes
            pred_xyxy = bx.xyxy.cpu().numpy().astype(np.float32)
            pred_cls  = bx.cls.cpu().numpy().astype(np.int32)
            pred_sc   = bx.conf.cpu().numpy().astype(np.float32)

        # Confusion matrix (match at IoU)
        matches, fp_idx, fn_idx = compute_matches(pred_xyxy, pred_cls, pred_sc, gt_xyxy, gt_cls, iou_thr=match_iou)

        # Count per-class
        for gi in range(len(gt_cls)):
            support[gt_cls[gi]] += 1

        for (pi, gi) in matches:
            cm[gt_cls[gi], pred_cls[pi]] += 1
            per_class_tp[gt_cls[gi]] += 1

        for pi in fp_idx:
            per_class_fp[pred_cls[pi]] += 1

        for gi in fn_idx:
            per_class_fn[gt_cls[gi]] += 1

        # ROC one-vs-rest: push every prediction score once, plus "implicit negative" for GT classes without a matching pred
        # Strategy: for each class c, scores list gets:
        #  - prediction scores for all boxes of class c (label=1 if matched to GT of class c else 0)
        #  - one zero-score for each FN of class c (to anchor negatives)
        matched_pred_indices = set(pi for pi,_ in matches)
        for c in range(num_classes):
            # predictions of class c
            idx_c = np.where(pred_cls==c)[0]
            for pi in idx_c:
                label = 1 if any((pi==mpi and gt_cls[mgi]==c) for (mpi,mgi) in matches) else 0
                roc_scores[c].append(float(pred_sc[pi]))
                roc_labels[c].append(label)
            # add zero-score negatives for each FN of class c
            fn_c = np.sum(gt_cls[fn_idx]==c) if len(fn_idx) else 0
            if fn_c>0:
                roc_scores[c].extend([0.0]*int(fn_c))
                roc_labels[c].extend([0]*int(fn_c))

        # Optional rendering
        if render:
            img = cv2.imread(path)
            if img is not None and len(pred_xyxy):
                img = render_boxes(img, pred_xyxy, pred_cls, pred_sc, class_names, box_thick, font_scale, font_thick)
            out_img = out_dir / "renders" / Path(path).name
            ensure_dir(out_img.parent)
            if img is not None:
                cv2.imwrite(str(out_img), img)

    # ---- Save confusion matrix image + CSV
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7,7), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(cm, cmap="Purples")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels([class_names[i] for i in range(num_classes)], rotation=45, ha="right")
    ax.set_yticklabels([class_names[i] for i in range(num_classes)])
    # numbers
    for i in range(num_classes):
        for j in range(num_classes):
            v = int(cm[i,j])
            if v>0:
                ax.text(j, i, str(v), va="center", ha="center", color="white" if v>np.max(cm)/2 else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png"); plt.close(fig)

    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # ---- Per-class metrics
    precision = np.divide(per_class_tp, (per_class_tp+per_class_fp), out=np.zeros_like(per_class_tp, dtype=float), where=(per_class_tp+per_class_fp)!=0)
    recall    = np.divide(per_class_tp, (per_class_tp+per_class_fn), out=np.zeros_like(per_class_tp, dtype=float), where=(per_class_tp+per_class_fn)!=0)
    f1        = np.divide(2*precision*recall, (precision+recall), out=np.zeros_like(precision), where=(precision+recall)!=0)

    import csv
    with open(out_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id","class_name","support","TP","FP","FN","precision","recall","F1"])
        for c in range(num_classes):
            w.writerow([c, class_names[c], int(support[c]), int(per_class_tp[c]), int(per_class_fp[c]), int(per_class_fn[c]),
                        round(float(precision[c]),6), round(float(recall[c]),6), round(float(f1[c]),6)])

    # ---- ROC & AUC
    # Build micro-averaged arrays by concatenation
    from sklearn.metrics import roc_curve, auc  # available in most envs; if not, we do manual

    roc_dir = ensure_dir(out_dir / "roc")
    aucs = {}
    # per-class
    for c in range(num_classes):
        y_true = np.array(roc_labels[c], dtype=int)
        y_score= np.array(roc_scores[c], dtype=float)
        if y_true.size < 2 or len(np.unique(y_true))<2:
            # not enough positives/negatives
            aucs[f"class_{c}"] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        AUC = auc(fpr, tpr)
        aucs[f"class_{c}"] = float(AUC)

        fig = plt.figure(figsize=(4,4), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, lw=1.5)
        ax.plot([0,1],[0,1],"--", lw=1, color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {class_names[c]} (AUC={AUC:.3f})")
        ax.grid(True, ls=":", lw=0.5)
        fig.tight_layout()
        fig.savefig(roc_dir / f"roc_{c:02d}_{class_names[c]}.png")
        plt.close(fig)

    # micro/macro
    y_true_micro = np.concatenate([np.array(roc_labels[c], int) for c in range(num_classes)]) if num_classes>0 else np.array([], int)
    y_score_micro= np.concatenate([np.array(roc_scores[c], float) for c in range(num_classes)]) if num_classes>0 else np.array([], float)
    if y_true_micro.size>0 and len(np.unique(y_true_micro))>=2:
        fpr_mi, tpr_mi, _ = roc_curve(y_true_micro, y_score_micro)
        aucs["micro"] = float(auc(fpr_mi, tpr_mi))
        fig = plt.figure(figsize=(4,4), dpi=200); ax = fig.add_subplot(111)
        ax.plot(fpr_mi, tpr_mi, lw=1.5, label=f"AUC={aucs['micro']:.3f}")
        ax.plot([0,1],[0,1],"--", color="gray"); ax.legend()
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC — micro average"); ax.grid(True, ls=":", lw=0.5)
        fig.tight_layout(); fig.savefig(roc_dir / f"roc_micro.png"); plt.close(fig)
    else:
        aucs["micro"] = float("nan")

    # macro = average of per-class AUC (ignoring NaN)
    cls_aucs = [v for k,v in aucs.items() if k.startswith("class_") and not math.isnan(v)]
    aucs["macro"] = float(np.mean(cls_aucs)) if len(cls_aucs) else float("nan")
    with open(out_dir / "roc_auc.json", "w", encoding="utf-8") as f:
        json.dump(aucs, f, indent=2)

    return {
        "cm": cm.tolist(),
        "per_class": {
            "support": support.tolist(),
            "tp": per_class_tp.tolist(),
            "fp": per_class_fp.tolist(),
            "fn": per_class_fn.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
        },
        "auc": aucs
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.45, help="NMS IoU for inference")
    ap.add_argument("--match_iou", type=float, default=0.50, help="IoU to match pred↔GT for CM/ROC")
    ap.add_argument("--render", type=str, default="false")
    ap.add_argument("--box_thick", type=int, default=6)
    ap.add_argument("--font_scale", type=float, default=1.0)
    ap.add_argument("--font_thick", type=int, default=2)
    args = ap.parse_args()

    render = str(args.render).strip().lower() in {"1","true","yes"}

    model = YOLO(args.weights)

    # read names from model (8 classes expected)
    names = model.model.names if hasattr(model, "model") else model.names
    if isinstance(names, dict):
        class_names = {int(k): v for k,v in names.items()}
    else:
        class_names = {i: n for i,n in enumerate(names)}

    out_dir = ensure_dir(Path(args.out_dir))
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    summary = {}
    for split in ["train", "val", "test"]:
        summary[split] = evaluate_split(
            model, args.data_yaml, split, out_dir, class_names,
            imgsz=args.imgsz, device=args.device, conf=args.conf, iou_nms=args.iou, match_iou=args.match_iou,
            render=render, box_thick=args.box_thick, font_scale=args.font_scale, font_thick=args.font_thick
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
