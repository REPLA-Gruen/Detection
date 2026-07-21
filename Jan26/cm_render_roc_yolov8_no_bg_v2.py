#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, json, math, csv, yaml, io, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# ---------- utilities

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
        color = (0, 170, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thick)
        tx = names.get(int(cid), str(cid))
        # write INSIDE the box (works even for full-frame)
        xx = max(int(x1)+6, 4)
        yy = max(int(y1)+18, 18)
        draw_label(img, f"{tx} {sc:.2f}", (xx, yy), font_scale, font_thick)
    return img

def xywhn_to_xyxy(box, w, h):
    x_c, y_c, bw, bh = box
    x1 = (x_c - bw/2) * w
    y1 = (y_c - bh/2) * h
    x2 = (x_c + bw/2) * w
    y2 = (y_c + bh/2) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def compute_matches(pred_xyxy, pred_cls, pred_scores, gt_xyxy, gt_cls, iou_thr=0.50):
    if len(pred_xyxy)==0 and len(gt_xyxy)==0:
        return [], np.array([], int), np.array([], int)

    ious = box_iou(
        torch.tensor(pred_xyxy, dtype=torch.float32),
        torch.tensor(gt_xyxy,   dtype=torch.float32)
    ).numpy() if (len(pred_xyxy) and len(gt_xyxy)) else np.zeros((len(pred_xyxy), len(gt_xyxy)), dtype=np.float32)

    matches = []
    used_gt = set()
    order = np.argsort(-pred_scores)
    for pi in order:
        if len(gt_xyxy)==0:
            break
        gi_cands = np.where(ious[pi] >= iou_thr)[0]
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

def expand_split_to_images(root_path: Path, split_entry):
    """
    split_entry may be:
      - a relative path to a folder with images
      - an absolute path
      - a .txt file listing image paths
    We return a list of image file paths (strings).
    """
    p = Path(split_entry)
    if not p.is_absolute():
        p = (root_path / p).resolve()

    if p.is_file() and p.suffix.lower()==".txt":
        with open(p, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
        return files
    elif p.is_dir():
        imgs = []
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
        for e in exts:
            imgs.extend([str(x) for x in p.rglob(f"*{e}")])
        return imgs
    else:
        # maybe a single image file was given
        if p.exists():
            return [str(p)]
        return []

# ---------- main evaluation routine

def evaluate_split_on_list(model, class_names, img_paths, match_iou, conf, iou_nms, render, out_dir,
                           box_thick, font_scale, font_thick):

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_tp = np.zeros(num_classes, dtype=np.int64)
    per_class_fp = np.zeros(num_classes, dtype=np.int64)
    per_class_fn = np.zeros(num_classes, dtype=np.int64)
    support      = np.zeros(num_classes, dtype=np.int64)

    roc_scores = [[] for _ in range(num_classes)]
    roc_labels = [[] for _ in range(num_classes)]

    renders_dir = ensure_dir(out_dir / "renders") if render else None

    for path in tqdm(img_paths, desc=f"[{out_dir.name}] images"):
        path = str(path)
        img = cv2.imread(path)
        if img is None:
            continue
        imh, imw = img.shape[:2]

        # ground truth
        lbl_path = Path(path).with_suffix(".txt").as_posix().replace("/images/", "/labels/").replace("\\images\\","\\labels\\")
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

        for g in gt_cls:
            support[g] += 1

        # predictions
        preds = model.predict(source=path, conf=conf, iou=iou_nms, imgsz=max(imh, imw), verbose=False)
        if not preds or preds[0].boxes is None or len(preds[0].boxes)==0:
            pred_xyxy = np.zeros((0,4), dtype=np.float32)
            pred_cls  = np.zeros((0,), dtype=np.int32)
            pred_sc   = np.zeros((0,), dtype=np.float32)
        else:
            bx = preds[0].boxes
            pred_xyxy = bx.xyxy.cpu().numpy().astype(np.float32)
            pred_cls  = bx.cls.cpu().numpy().astype(np.int32)
            pred_sc   = bx.conf.cpu().numpy().astype(np.float32)

        matches, fp_idx, fn_idx = compute_matches(pred_xyxy, pred_cls, pred_sc, gt_xyxy, gt_cls, iou_thr=match_iou)

        for (pi, gi) in matches:
            cm[gt_cls[gi], pred_cls[pi]] += 1
            per_class_tp[gt_cls[gi]] += 1
        for pi in fp_idx:
            per_class_fp[pred_cls[pi]] += 1
        for gi in fn_idx:
            per_class_fn[gt_cls[gi]] += 1

        # ROC one-vs-rest
        matched_pred_indices = set(pi for pi,_ in matches)
        for c in range(num_classes):
            idx_c = np.where(pred_cls==c)[0]
            for pi in idx_c:
                label = 1 if any((pi==mpi and gt_cls[mgi]==c) for (mpi,mgi) in matches) else 0
                roc_scores[c].append(float(pred_sc[pi]))
                roc_labels[c].append(label)
            fn_c = np.sum(gt_cls[fn_idx]==c) if len(fn_idx) else 0
            if fn_c>0:
                roc_scores[c].extend([0.0]*int(fn_c))
                roc_labels[c].extend([0]*int(fn_c))

        if render:
            img_r = img.copy()
            if len(pred_xyxy):
                img_r = render_boxes(img_r, pred_xyxy, pred_cls, pred_sc, class_names, box_thick, font_scale, font_thick)
            out_img = renders_dir / Path(path).name
            cv2.imwrite(str(out_img), img_r)

    # save CM
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7,7), dpi=200)
    ax = fig.add_subplot(111)
    ax.imshow(cm, cmap="Purples")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels([class_names[i] for i in range(num_classes)], rotation=45, ha="right")
    ax.set_yticklabels([class_names[i] for i in range(num_classes)])
    for i in range(num_classes):
        for j in range(num_classes):
            v = int(cm[i,j])
            if v>0:
                ax.text(j, i, str(v), va="center", ha="center",
                        color=("white" if v>np.max(cm)/2 else "black"), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png"); plt.close(fig)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # per-class metrics
    precision = np.divide(per_class_tp, (per_class_tp+per_class_fp), out=np.zeros_like(per_class_tp, dtype=float), where=(per_class_tp+per_class_fp)!=0)
    recall    = np.divide(per_class_tp, (per_class_tp+per_class_fn), out=np.zeros_like(per_class_tp, dtype=float), where=(per_class_tp+per_class_fn)!=0)
    f1        = np.divide(2*precision*recall, (precision+recall), out=np.zeros_like(precision), where=(precision+recall)!=0)

    with open(out_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id","class_name","support","TP","FP","FN","precision","recall","F1"])
        for c in range(num_classes):
            w.writerow([c, class_names[c], int(support[c]), int(per_class_tp[c]), int(per_class_fp[c]), int(per_class_fn[c]),
                        round(float(precision[c]),6), round(float(recall[c]),6), round(float(f1[c]),6)])

    # ROC & AUC
    from sklearn.metrics import roc_curve, auc
    roc_dir = ensure_dir(out_dir / "roc")
    aucs = {}
    for c in range(num_classes):
        y_true = np.array(roc_labels[c], dtype=int)
        y_score= np.array(roc_scores[c], dtype=float)
        if y_true.size < 2 or len(np.unique(y_true))<2:
            aucs[f"class_{c}"] = float("nan"); continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        AUC = auc(fpr, tpr); aucs[f"class_{c}"] = float(AUC)
        fig = plt.figure(figsize=(4,4), dpi=200); ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, lw=1.5); ax.plot([0,1],[0,1],"--", color="gray")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — {class_names[c]} (AUC={AUC:.3f})"); ax.grid(True, ls=":", lw=0.5)
        fig.tight_layout(); fig.savefig(roc_dir / f"roc_{c:02d}_{class_names[c]}.png"); plt.close(fig)

    cls_aucs = [v for k,v in aucs.items() if k.startswith("class_") and not math.isnan(v)]
    aucs["macro"] = float(np.mean(cls_aucs)) if cls_aucs else float("nan")
    # micro
    y_true_micro = np.concatenate([np.array(roc_labels[c], int) for c in range(num_classes)]) if num_classes else np.array([], int)
    y_score_micro= np.concatenate([np.array(roc_scores[c], float) for c in range(num_classes)]) if num_classes else np.array([], float)
    if y_true_micro.size>1 and len(np.unique(y_true_micro))>=2:
        fpr_m, tpr_m, _ = roc_curve(y_true_micro, y_score_micro)
        aucs["micro"] = float(auc(fpr_m, tpr_m))
        fig = plt.figure(figsize=(4,4), dpi=200); ax = fig.add_subplot(111)
        ax.plot(fpr_m, tpr_m, lw=1.5, label=f"AUC={aucs['micro']:.3f}")
        ax.plot([0,1],[0,1],"--", color="gray"); ax.legend(); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC — micro"); ax.grid(True, ls=":", lw=0.5)
        fig.tight_layout(); fig.savefig(roc_dir / "roc_micro.png"); plt.close(fig)
    else:
        aucs["micro"] = float("nan")

    with open(out_dir / "roc_auc.json", "w", encoding="utf-8") as f:
        json.dump(aucs, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.45, help="NMS IoU")
    ap.add_argument("--match_iou", type=float, default=0.50, help="IoU for pred↔GT match")
    ap.add_argument("--render", type=str, default="false")
    ap.add_argument("--box_thick", type=int, default=6)
    ap.add_argument("--font_scale", type=float, default=1.0)
    ap.add_argument("--font_thick", type=int, default=2)
    args = ap.parse_args()

    render = str(args.render).strip().lower() in {"1","true","yes"}
    model = YOLO(args.weights)
    if args.device is not None:
        model.to(args.device)

    # class names
    names = model.model.names if hasattr(model, "model") else model.names
    class_names = {int(k): v for k,v in (names.items() if isinstance(names, dict) else {i:n for i,n in enumerate(names)}.items())}

    # dataset YAML
    with open(args.data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path","."))
    splits = {
        "train": y["train"],
        "val":   y["val"],
        "test":  y["test"]
    }

    out_root = ensure_dir(Path(args.out_dir))
    with open(out_root / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    for split, entry in splits.items():
        img_list = expand_split_to_images(root, entry)
        if len(img_list)==0:
            print(f"[WARN] No images found for split '{split}' from entry: {entry}")
            continue
        split_dir = ensure_dir(out_root / split)
        evaluate_split_on_list(
            model, class_names, img_list, args.match_iou, args.conf, args.iou, render,
            split_dir, args.box_thick, args.font_scale, args.font_thick
        )

if __name__ == "__main__":
    main()
