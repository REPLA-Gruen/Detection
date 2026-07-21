# -*- coding: utf-8 -*-
"""
Render YOLO labels (8 classes) back onto images, preserving folders.
"""

import argparse
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

TARGET_CLASSES = [
    "BAW-Sack","Plastic","Metal","Glass",
    "Paper_Cardboard","Other","Wagon","Container"
]

def draw(bgr, yolo_path, box_thick, font_scale, font_thick):
    H,W = bgr.shape[:2]
    out = bgr.copy()
    if not yolo_path.exists(): return out
    for line in yolo_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts)!=5: continue
        cls = int(parts[0]); cx,cy,bw,bh = map(float, parts[1:])
        x1 = int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
        x2 = int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
        color = tuple(int((hash(TARGET_CLASSES[cls])>>s)&255) for s in (16,8,0))
        label = TARGET_CLASSES[cls]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,box_thick)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,font_scale,font_thick)
        y1t = max(th+4,y1)
        cv2.rectangle(out,(x1,y1t-th-4),(x1+tw+6,y1t),color,-1)
        cv2.putText(out,label,(x1+3,y1t-3),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(255,255,255),font_thick,cv2.LINE_AA)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", required=True)  # same root used above
    ap.add_argument("--split", default="train")    # train or val
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--box_thick", type=int, default=6)
    ap.add_argument("--font_scale", type=float, default=0.9)
    ap.add_argument("--font_thick", type=int, default=2)
    args = ap.parse_args()

    src_img_dir = Path(args.yolo_root)/f"images/{args.split}"
    src_lbl_dir = Path(args.yolo_root)/f"labels/{args.split}"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ims = list(src_img_dir.glob("*.*"))
    for ip in tqdm(ims, desc="Render YOLO"):
        bgr = cv2.imread(str(ip))
        if bgr is None: continue
        lbl = src_lbl_dir / (ip.stem + ".txt")
        vis = draw(bgr, lbl, args.box_thick, args.font_scale, args.font_thick)
        cv2.imwrite(str(out_dir/ip.name), vis)

if __name__ == "__main__":
    main()
