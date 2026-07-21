# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, shutil, sys, signal
from pathlib import Path
from typing import List, Tuple, Optional
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ========= visuals (NEW defaults for thicker/bigger drawing) =========
BOX_THICK   = 15     # 8 rectangle border thickness
FONT_SCALE  = 3   # 1.4 label font scale (bigger)
FONT_THICK  = 12     # 3 label font thickness (bolder)

# --------- default paths (your run) ----------
RUN_DIR   = Path(r"C:\Hadi\Foreign_Material\Runs\yolov8n_Ours_retrainfrom_AllDSs_Full_Trained")
WEIGHTS   = RUN_DIR / "weights" / "best.pt"
DATA_YAML = Path(r"C:\Hadi\Foreign_Material\Code\Python\tools\dataset_ours.yml")
SAVE_ROOT = RUN_DIR
OUT_DIR_PRED = SAVE_ROOT / "test_preds_colored"
OUT_DIR_EVAL = SAVE_ROOT / "eval_test"

# --------- classes & colors ----------
CLASS_NAMES = ["BAW-Sack","Plastic","Metal","Glass","Paper_Cardboard","Other"]
PALETTE = {
    "BAW-Sack": (0,255,255),        # yellow (BGR)
    "Plastic": (255,0,0),           # blue
    "Metal": (0,165,255),           # orange
    "Glass": (0,255,0),             # green
    "Paper_Cardboard": (255,191,0), # cyan-ish
    "Other": (255,0,255),           # magenta
}
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

# Optional tqdm (fallback to prints if missing)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- helpers ----------
def _resolve_split_paths(yaml_file: Path, override: Optional[str]) -> Tuple[List[Path], Optional[Path]]:
    y = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    base = Path(y.get("path", yaml_file.parent)).resolve()
    split = override or y.get("test") or y.get("val")
    if split is None:
        raise FileNotFoundError("No 'test' or 'val' in YAML and no override provided.")

    def to_files(spec) -> List[Path]:
        files: List[Path] = []
        items = spec if isinstance(spec, list) else [spec]
        for it in items:
            p = Path(it)
            if not p.is_absolute():
                p = base / it
            if p.is_dir():
                files += [f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS]
            elif p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
            else:
                files += [f for f in base.glob(str(p)) if f.suffix.lower() in IMG_EXTS]
        return sorted(set(files))

    imgs = to_files(split)

    # infer labels dir if possible
    labels_dir = None
    if imgs:
        first = imgs[0]
        parts = list(first.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            labels_dir = Path(*parts[:-1])
            if not labels_dir.exists():
                labels_dir = None

    if not imgs:
        raise FileNotFoundError(
            f"No test images resolved.\n  YAML: {yaml_file}\n  base: {base}\n  split: {split}"
        )
    return imgs, labels_dir

def _draw_one(im: np.ndarray, b: np.ndarray, c: int, conf: float) -> None:
    x1,y1,x2,y2 = map(int, b)
    cls_name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else f"cls{c}"
    color = PALETTE.get(cls_name, (200,200,200))

    # Thick rectangle
    cv2.rectangle(im, (x1,y1), (x2,y2), color, BOX_THICK)

    # Label text + background strip (uses new FONT_SCALE / FONT_THICK)
    label = f"{cls_name} {conf:.2f}"
    (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(im, (x1, y_top), (x1 + tw + 10, y_top + th + 8), color, -1)
    cv2.putText(
        im, label, (x1 + 5, y_top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), FONT_THICK, cv2.LINE_AA
    )

def _xywhn_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = box
    return np.array([(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h], dtype=np.float32)

def _load_gt(lbl_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    if not lbl_path or not lbl_path.exists():
        return np.empty((0,4), np.float32), np.empty((0,), np.int32)
    rows = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        c = int(float(parts[0]))
        box = np.array(list(map(float, parts[1:])))
        rows.append((c, _xywhn_to_xyxy(box, img_w, img_h)))
    if not rows:
        return np.empty((0,4), np.float32), np.empty((0,), np.int32)
    cls = np.array([r[0] for r in rows], np.int32)
    bxs = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
    return bxs, cls

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a)==0 or len(b)==0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:,2]-a[:,0]).clip(0) * (a[:,3]-a[:,1]).clip(0)
    area_b = (b[:,2]-b[:,0]).clip(0) * (b[:,3]-b[:,1]).clip(0)
    ious = np.zeros((len(a), len(b)), dtype=np.float32)
    for i in range(len(a)):
        x1 = np.maximum(a[i,0], b[:,0])
        y1 = np.maximum(a[i,1], b[:,1])
        x2 = np.minimum(a[i,2], b[:,2])
        y2 = np.minimum(a[i,3], b[:,3])
        inter = (x2-x1).clip(0) * (y2-y1).clip(0)
        union = area_a[i] + area_b - inter + 1e-9
        ious[i,:] = inter / union
    return ious

def _update_confusion_6x6(confmat: np.ndarray,
                          gt_b: np.ndarray, gt_c: np.ndarray,
                          pr_b: np.ndarray, pr_c: np.ndarray,
                          iou_thres: float = 0.5) -> None:
    if len(gt_b)==0 or len(pr_b)==0:
        return
    ious = _iou_matrix(gt_b, pr_b)
    used_pred = set()
    for i in range(len(gt_b)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thres and j not in used_pred:
            gcls = int(gt_c[i])
            pcls = int(pr_c[j])
            if 0 <= gcls < confmat.shape[0] and 0 <= pcls < confmat.shape[1]:
                confmat[gcls, pcls] += 1
            used_pred.add(j)

def _plot_conf(mat: np.ndarray, fn: Path, norm: bool, names: list[str]):
    plt.figure(figsize=(9,7))
    show = mat / (mat.sum(1, keepdims=True) + 1e-9) if norm else mat
    im = plt.imshow(show, cmap="Blues")
    plt.colorbar(im)
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{show[i,j]:.2f}" if norm else f"{int(show[i,j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color="k")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix (Normalized)" if norm else "Confusion Matrix")
    plt.tight_layout(); plt.savefig(fn, dpi=160); plt.close()

def _save_partial(conf6: np.ndarray):
    OUT_DIR_EVAL.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR_EVAL / "confusion6_raw.npy", conf6)
    _plot_conf(conf6, OUT_DIR_EVAL / "confusion6.png", norm=False, names=CLASS_NAMES)
    _plot_conf(conf6, OUT_DIR_EVAL / "confusion6_normalized.png", norm=True, names=CLASS_NAMES)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--override_test", default="")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--iou_cm", type=float, default=0.5, help="IoU threshold for confusion matching")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)   # Windows-safe
    ap.add_argument("--batch", type=int, default=8)     # final val()
    args = ap.parse_args()

    # Prepare dirs
    OUT_DIR_PRED.mkdir(parents=True, exist_ok=True)
    OUT_DIR_EVAL.mkdir(parents=True, exist_ok=True)

    imgs, labels_dir = _resolve_split_paths(Path(args.data), args.override_test or None)
    total = len(imgs)
    print(f"[eval] Found {total} test images")
    print(f"[eval] Labels dir: {labels_dir if labels_dir else '(none found)'}")
    print(f"[eval] Weights: {args.weights}")
    print(f"[eval] Renders -> {OUT_DIR_PRED}")
    print(f"[eval] Metrics/plots -> {OUT_DIR_EVAL}")

    # Clean renders directory for a fresh pass
    if OUT_DIR_PRED.exists():
        shutil.rmtree(OUT_DIR_PRED)
    OUT_DIR_PRED.mkdir(parents=True, exist_ok=True)

    # Model
    print("[eval] Loading model...")
    model = YOLO(args.weights)

    # Confusion
    conf6 = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

    interrupted = {"flag": False}
    def _handle_sigint(sig, frame):
        interrupted["flag"] = True
        print("\n[eval] Interrupted — saving partial results...")
        _save_partial(conf6)
        sys.exit(130)
    signal.signal(signal.SIGINT, _handle_sigint)

    iterator = imgs if tqdm is None else tqdm(imgs, desc="Rendering & updating confusion", unit="img", ncols=80)

    # Stream: one image at a time (low RAM)
    for img_path in iterator:
        for res in model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            stream=True,
            save=False,
            verbose=False,
        ):
            im = res.orig_img.copy()
            h, w = im.shape[:2]

            # predictions
            if res and len(res.boxes) > 0:
                pr_b = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                pr_c = res.boxes.cls.cpu().numpy().astype(np.int32)
                pr_s = res.boxes.conf.cpu().numpy().astype(np.float32)
                for bb, cc, ss in zip(pr_b, pr_c, pr_s):
                    _draw_one(im, bb, int(cc), float(ss))
            else:
                pr_b = np.empty((0,4), np.float32)
                pr_c = np.empty((0,), np.int32)

            # save rendered
            OUT_DIR_PRED.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT_DIR_PRED / Path(res.path).name), im)

            # update confusion
            if labels_dir:
                lbl = labels_dir / (Path(res.path).stem + ".txt")
                gt_b, gt_c = _load_gt(lbl, w, h)
                _update_confusion_6x6(conf6, gt_b, gt_c, pr_b, pr_c, iou_thres=args.iou_cm)

    # Save confusion + YOLO val metrics
    print("[eval] Saving confusion matrices...")
    _save_partial(conf6)

    print("[eval] Running YOLO built-in test metrics (batched)...")
    val_res = model.val(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        device=args.device,
        split="test",
        workers=args.workers,
        batch=args.batch
    )
    (OUT_DIR_EVAL / "metrics.json").write_text(json.dumps(val_res.results_dict, indent=2))

    print(f"[eval] Done.\n  Renders: {OUT_DIR_PRED}\n  Metrics: {OUT_DIR_EVAL}")

if __name__ == "__main__":
    main()
