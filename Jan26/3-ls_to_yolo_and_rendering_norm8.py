# -*- coding: utf-8 -*-
"""
Label Studio export  ->  YOLOv8 DET (8-class normalized schema)
Keeps only images that have at least one valid object after mapping/geometry.
Preserves your existing split by reading from YOLO-style folders:
  <root1>/train/images, <root2>/val/images, <root3>/test/images   (semicolon-separated)

Classes (index -> name):
  0: BAW-Sack
  1: Plastic
  2: Metal
  3: Glass
  4: Paper_Cardboard
  5: Other
  6: Wagon
  7: Container

Usage (PowerShell):
    python C:\Hadi\Foreign_Material\Code\Python\tools\3-ls_to_yolo_and_rendering_norm8.py `
      --export_json  "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\project-2-at-2026-01-06-10-09-5c186fe2.json" `
      --image_roots  "C:\Hadi\Foreign_Material\Data\Ours\yolo\train\images;C:\Hadi\Foreign_Material\Data\Ours\yolo\val\images;C:\Hadi\Foreign_Material\Data\Ours\yolo\test\images" `
      --yolo_root    "C:\Hadi\Foreign_Material\Data\Ours\Jan-26\YOLO_norm8" `
      --prefix       "OUR__" `
      --render       true `
      --box_thick 10 --font_scale 1.5 --font_thick 5

"""

import argparse, json, shutil, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from tqdm import tqdm
import random

CLASSES = ["BAW-Sack","Plastic","Metal","Glass","Paper_Cardboard","Other","Wagon","Container"]
NAME2ID = {n:i for i,n in enumerate(CLASSES)}

FONT = cv2.FONT_HERSHEY_SIMPLEX

def put_center_label(img, x, y, w, h, text, color, font_scale=0.9, font_thick=2, alpha=0.35, pad=6):
    # text size
    tsize = cv2.getTextSize(text, FONT, font_scale, font_thick)[0]
    tx = int(x + (w - tsize[0]) / 2)
    ty = int(y + (h + tsize[1]) / 2)

    # clamp to image
    H, W = img.shape[:2]
    tx = max(0, min(W - tsize[0] - 1, tx))
    ty = max(tsize[1] + 1, min(H - 1, ty))

    # semi-transparent box behind text
    overlay = img.copy()
    cv2.rectangle(overlay, (tx - pad, ty - tsize[1] - pad),
                           (tx + tsize[0] + pad, ty + pad), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # foreground text
    cv2.putText(img, text, (tx, ty), FONT, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def color_for(name: str) -> Tuple[int,int,int]:
    s = abs(hash(name)) % (256**3)
    return ((s>>16)&255, (s>>8)&255, s&255)

# ---------- robust LS geometry readers ----------
def rect_percent_to_abs(value: dict, H: int, W: int) -> Tuple[int,int,int,int]:
    ow = int(value.get("original_width",  W) or W)
    oh = int(value.get("original_height", H) or H)
    x  = float(value.get("x",0)); y = float(value.get("y",0))
    w  = float(value.get("width",0)); h = float(value.get("height",0))
    # LS rectangles are usually in percent of original image (0..100)
    if max(x,y,w,h) <= 100.0:
        px = round(x*ow/100.0); py = round(y*oh/100.0)
        pw = round(w*ow/100.0);  ph = round(h*oh/100.0)
    else:  # already in px but relative to original dims
        px = round(x*W/ow); py = round(y*H/oh)
        pw = round(w*W/ow); ph = round(h*H/oh)
    px = int(max(0, min(px, W-1))); py = int(max(0, min(py, H-1)))
    pw = int(max(1, min(pw, W-px))); ph = int(max(1, min(ph, H-py)))
    return (px,py,pw,ph)

def polygon_to_mask(points: List[List[float]], H: int, W: int, oh: int, ow: int) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    # LS polygons are in percents (0..100) of original dims
    if pts.max() <= 100.0:
        pts[:,0] = pts[:,0] * ow / 100.0
        pts[:,1] = pts[:,1] * oh / 100.0
    pts = np.round(pts).astype(np.int32)
    m = np.zeros((oh, ow), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 255)
    if (oh,ow) != (H,W):
        m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)
    return m

# Minimal, robust RLE decoder for LS "brushlabels" masks stored as bytes array
def _access_bit(data, num):
    base = int(num//8); shift = 7 - int(num%8)
    return (data[base] & (1<<shift)) >> shift

def _bytes2bit(data: List[int]) -> str:
    return "".join([str(_access_bit(data, i)) for i in range(len(data)*8)])

class _In:
    def __init__(self, bits): self.bits = bits; self.i = 0
    def read(self, n): out = self.bits[self.i:self.i+n]; self.i += n; return int(out,2)

def rle_to_mask(rle: List[int], oh: int, ow: int) -> np.ndarray:
    bits = _bytes2bit(rle); s = _In(bits)
    num  = s.read(32); word = s.read(5)+1
    sizes = [s.read(4)+1 for _ in range(4)]
    i = 0; out = np.zeros(num, dtype=np.uint8)
    while i < num:
        flag = s.read(1)
        j = i + 1 + s.read(sizes[s.read(2)])
        if flag:
            val = s.read(word); out[i:j] = val; i = j
        else:
            while i < j:
                val = s.read(word); out[i] = val; i += 1
    img = np.reshape(out,[oh,ow,4])[:,:,3]
    return (img>0).astype(np.uint8)

def mask_to_box(m: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    cnts = cv2.findContours((m*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts)==2 else cnts[1]
    if not cnts: return None
    x,y,w,h = cv2.boundingRect(np.vstack(cnts))
    return (x,y,w,h)

# ---------- label mapping (8-class) ----------
# We merge LS raw labels into 8 final classes using simple, transparent rules.
# (You asked to explicitly state this.)
PLASTIC_HINTS = ("plast","kunststoff","poly","pet","tüte","tute","beutel","bag","sack")
METAL_HINTS   = ("metal","metall","alu","alube","can","dose")
GLASS_HINTS   = ("glas","glass","bottle","flasche")
PAPER_HINTS   = ("paper","papier","pappe","card","karton","cardboard")
WAGON_HINTS   = ("wagon","waggon","wagen")
CONTAINER_HINTS = ("container","bin","mullcontainer","müllcontainer")

def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_\-]+","",s)
    return s

def map_to_norm8(raw: str) -> int:
    s = norm(raw)
    if "baw" in s:                 return NAME2ID["BAW-Sack"]
    if any(k in s for k in PLASTIC_HINTS):  return NAME2ID["Plastic"]
    if any(k in s for k in METAL_HINTS):    return NAME2ID["Metal"]
    if any(k in s for k in GLASS_HINTS):    return NAME2ID["Glass"]
    if any(k in s for k in PAPER_HINTS):    return NAME2ID["Paper_Cardboard"]
    if any(k in s for k in WAGON_HINTS):    return NAME2ID["Wagon"]
    if any(k in s for k in CONTAINER_HINTS):return NAME2ID["Container"]
    # default bucket
    return NAME2ID["Other"]

# ---------- helpers ----------
def build_image_index(roots: List[Path]) -> Dict[str, Tuple[Path,str]]:
    """
    Return {lower_basename: (full_path, split)} where split is train/val/test inferred from folder name.
    """
    idx = {}
    for r in roots:
        split = r.parent.name.lower()  # images/<split> -> take parent
        for p in r.rglob("*.*"):
            if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tif",".tiff"):
                idx[p.name.lower()] = (p, split)
    return idx

def yolo_line(cx,cy,w,h,H,W):
    return f"{cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_json", required=True)
    ap.add_argument("--image_roots", required=True,
                    help="Semicolon-separated list of existing '.../train/images;.../val/images;.../test/images'")
    ap.add_argument("--yolo_root", required=True)
    ap.add_argument("--prefix", default="OUR__")
    ap.add_argument("--render", default="false")
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument("--box_thick", type=int, default=6)
    ap.add_argument("--font_scale", type=float, default=0.9)
    ap.add_argument("--font_thick", type=int, default=2)
    args = ap.parse_args()

    do_render = str(args.render).lower() in ("1","true","yes","y")

    yroot = Path(args.yolo_root)
    for split in ("train","val","test"):
        ensure_dir(yroot/"images"/split)
        ensure_dir(yroot/"labels"/split)
        if do_render:
            ensure_dir(yroot/"renders"/split)

    # index existing images
    roots = [Path(x) for x in args.image_roots.split(";") if x.strip()]
    idx = build_image_index(roots)
    if not idx:
        raise SystemExit(f"[fatal] No images were found under: {args.image_roots}")

    # read LS JSON
    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("[fatal] export_json must contain a JSON list")

    # group by basename (case-insensitive)
    groups: Dict[str, List[dict]] = {}
    for it in tasks:
        data = it.get("data",{}) or {}
        url  = data.get("image") or data.get("Image") or ""
        base = Path(url).name.lower() if url else ""
        if base:
            groups.setdefault(base, []).append(it)

    kept_imgs = 0
    kept_boxes = 0
    missing = 0
    unlabeled = 0
    bad_geom = 0

    log_lines = []
    per_class_boxes = {n:0 for n in CLASSES}

    for base, items in tqdm(groups.items(), desc="LS→YOLO(8) from existing"):
        if base not in idx:
            missing += 1
            log_lines.append(f"MISS\t{base}\tno-source-image")
            continue

        src_path, split = idx[base]
        dst_img = (yroot/"images"/split/(args.prefix + src_path.name))
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)

        img = cv2.imread(str(dst_img))
        if img is None:
            log_lines.append(f"READFAIL\t{base}")
            continue
        H,W = img.shape[:2]
        render = img.copy()

        ylines: List[str] = []

        # Iterate all results across annotations (GT + manual)
        for it in items:
            for ann in (it.get("annotations") or []):
                for res in (ann.get("result") or []):
                    val = res.get("value",{}) or {}

                    # extract raw label
                    raw = None
                    for k in ("labels","rectanglelabels","polygonlabels","brushlabels"):
                        v = val.get(k)
                        if isinstance(v, list) and v:
                            raw = str(v[0]); break
                    if not raw:
                        # purely classification/choice — skip
                        continue

                    cid = map_to_norm8(raw)
                    cname = CLASSES[cid]
                    col = color_for(cname)

                    oh = int(res.get("original_height", H) or H)
                    ow = int(res.get("original_width",  W) or W)

                    box = None
                    # Brush mask (RLE)
                    if "rle" in val and isinstance(val.get("rle"), list):
                        try:
                            m = rle_to_mask(val["rle"], oh, ow)
                            if (oh,ow) != (H,W):
                                m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)
                            box = mask_to_box(m)
                        except Exception:
                            box = None
                    # Polygon
                    elif "points" in val and isinstance(val["points"], list) and len(val["points"]) >= 3:
                        try:
                            m = polygon_to_mask(val["points"], H, W, oh, ow)
                            box = mask_to_box(m)
                        except Exception:
                            box = None
                    # Rectangle
                    elif {"x","y","width","height"}.issubset(val.keys()):
                        box = rect_percent_to_abs(val, H, W)

                    if box is None:
                        bad_geom += 1
                        continue

                    x,y,w,h = box
                    if w*h < args.min_area:
                        bad_geom += 1
                        continue

                    cx = x + w/2.0; cy = y + h/2.0
                    ylines.append(f"{cid} " + yolo_line(cx,cy,w,h,H,W))
                    kept_boxes += 1; per_class_boxes[cname] += 1
                    if do_render:
                        cv2.rectangle(render, (x, y), (x + w, y + h), col, args.box_thick)
                        label = cname
                    
                        if cname in ("Wagon", "Container"):
                            # draw the class name INSIDE the box (centered)
                            put_center_label(render, x, y, w, h, label, col,
                                             font_scale=args.font_scale, font_thick=args.font_thick)
                        else:
                            # original top-of-box label for other classes
                            t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, args.font_thick)[0]
                            bx1, by1 = x, max(0, y - t[1] - 6)
                            bx2, by2 = x + t[0] + 12, y
                            cv2.rectangle(render, (bx1, by1), (bx2, by2), col, -1)
                            cv2.putText(render, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        args.font_scale, (255, 255, 255), args.font_thick, cv2.LINE_AA)


                    # if do_render:
                    #     cv2.rectangle(render, (x,y), (x+w,y+h), col, args.box_thick)
                    #     label = cname
                    #     t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, args.font_thick)[0]
                    #     bx1,by1 = x, max(0, y - t[1] - 6)
                    #     bx2,by2 = x + t[0] + 12, y
                    #     cv2.rectangle(render, (bx1,by1), (bx2,by2), col, -1)
                    #     cv2.putText(render, label, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    #                 args.font_scale, (255,255,255), args.font_thick, cv2.LINE_AA)

        if ylines:
            (yroot/"labels"/split/(dst_img.stem+".txt")).write_text("\n".join(ylines), encoding="utf-8")
            if do_render:
                cv2.imwrite(str(yroot/"renders"/split/(dst_img.stem+".jpg")), render)
            kept_imgs += 1
        else:
            unlabeled += 1
            # keep image only if it already exists; otherwise remove copied image to keep dataset clean
            try:
                if (yroot/"labels"/split/(dst_img.stem+".txt")).exists() is False:
                    dst_img.unlink(missing_ok=True)
            except Exception:
                pass

    # YAML
    (yroot/"dataset_norm8.yaml").write_text(
        "\n".join([
            f"path: {yroot.as_posix()}",
            "train: images/train",
            "val:   images/val",
            "test:  images/test",
            "names:",
            *[f"  {i}: {n}" for i,n in enumerate(CLASSES)]
        ]),
        encoding="utf-8"
    )

    # logs
    (yroot/"convert_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    (yroot/"debug_counts.txt").write_text(
        json.dumps({
            "kept_imgs": kept_imgs,
            "kept_boxes": kept_boxes,
            "missing_images": missing,
            "unlabeled_images": unlabeled,
            "bad_geometry": bad_geom,
            "per_class_boxes": per_class_boxes
        }, indent=2),
        encoding="utf-8"
    )

    print("\n[summary]")
    print("  YOLO root     :", yroot)
    print("  Kept images   :", kept_imgs)
    print("  Kept boxes    :", kept_boxes)
    print("  Missing imgs  :", missing)
    print("  Unlabeled     :", unlabeled)
    print("  Bad geometry  :", bad_geom)
    print("  YAML          :", yroot/"dataset_norm8.yaml")
    print("  Logs          :", yroot/"convert_log.txt")
    print("  Classes (8)   :", CLASSES)

if __name__ == "__main__":
    main()
