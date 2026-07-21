#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Label Studio export -> YOLOv8 DET (6 detection classes) + Scene tag from annotations

Detection classes (index -> name):
  0: BAW-Sack
  1: Plastic
  2: Metal
  3: Glass
  4: Paper_Cardboard
  5: Other

Scene classes (not exported to YOLO labels; used only for rendering/metadata):
  Scene_Wagon, Scene_Container, Scene_Unknown

Images are taken ONLY from existing roots:
  <root>/train/images; <root>/val/images; <root>/test/images

a clean, working converter that does exactly what you asked: 6 detection classes + a separate scene tag (Container / Wagon / Unknown) derived only from the annotations, and renders the class name inside each box (so even huge “whole-image” boxes are readable). I’ve also fixed the CLI flag you tripped on.

"""
import argparse, json, os, pathlib, cv2
from collections import defaultdict

DETECT_CLASSES = ["BAW-Sack","Plastic","Metal","Glass","Paper_Cardboard","Other"]
ALIASES = {
    "paper/cardboard":"Paper_Cardboard","paper-cardboard":"Paper_Cardboard",
    "paper_cardboard":"Paper_Cardboard","baw_sack":"BAW-Sack","bawsack":"BAW-Sack",
}
def norm(s:str)->str: return (s or "").strip().lower().replace(" ","").replace("_","")
DET_IDX = { norm(k):i for i,k in enumerate(DETECT_CLASSES) }

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def list_images_under(*roots):
    exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}; m={}
    for root in roots:
        if not root: continue
        for p in pathlib.Path(root).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                m[p.name.lower()] = str(p)
    return m
def split_of_path(p, tr, va, te):
    p=os.path.normpath(p).lower()
    if tr and os.path.normpath(tr).lower() in p: return "train"
    if va and os.path.normpath(va).lower() in p: return "val"
    if te and os.path.normpath(te).lower() in p: return "test"
    for k in ("\\train\\","/train/"): 
        if k in p: return "train"
    for k in ("\\val\\","/val/"): 
        if k in p: return "val"
    for k in ("\\test\\","/test/"): 
        if k in p: return "test"
    return "train"

def clip_box(x1,y1,x2,y2,W,H):
    x1=max(0,min(x1,W-1)); y1=max(0,min(y1,H-1))
    x2=max(0,min(x2,W-1)); y2=max(0,min(y2,H-1))
    if x2<=x1 or y2<=y1: return None
    return x1,y1,x2,y2
def to_yolo(x1,y1,x2,y2,W,H):
    cx=(x1+x2)/2/W; cy=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
    return cx,cy,w,h

def _extract_results(task):
    if isinstance(task.get("annotations"),list) and task["annotations"]:
        ann=task["annotations"][0]
        if isinstance(ann,dict) and isinstance(ann.get("result"),list):
            return ann["result"]
    if isinstance(task.get("completions"),list) and task["completions"]:
        comp=task["completions"][0]
        if isinstance(comp,dict) and isinstance(comp.get("result"),list):
            return comp["result"]
    return []

def parse_labelstudio_annos(task, W=None, H=None):
    out=[]
    for r in _extract_results(task):
        v=r.get("value",{})
        labels = v.get("rectanglelabels") or v.get("labels") or v.get("choices") or []
        cname = labels[0] if isinstance(labels,list) and labels else (r.get("label") or v.get("label") or "")
        if {"x","y","width","height"} <= set(v.keys()) and W is not None and H is not None:
            x=float(v["x"])*W/100.0; y=float(v["y"])*H/100.0
            w=float(v["width"])*W/100.0; h=float(v["height"])*H/100.0
            out.append({"class_name":cname,"x1":int(x),"y1":int(y),"x2":int(x+w),"y2":int(y+h)}); continue
        pts=v.get("points")
        if isinstance(pts,list) and pts and W is not None and H is not None:
            xs=[float(p[0])*W/100.0 for p in pts]; ys=[float(p[1])*H/100.0 for p in pts]
            out.append({"class_name":cname,"x1":int(min(xs)),"y1":int(min(ys)),
                        "x2":int(max(xs)),"y2":int(max(ys))}); continue
        if {"x1","y1","x2","y2"} <= set(v.keys()):
            out.append({"class_name":cname,"x1":int(v["x1"]),"y1":int(v["y1"]),
                        "x2":int(v["x2"]),"y2":int(v["y2"])})
    return out

def scene_from_annos(annos):
    names=[norm(a.get("class_name","")) for a in annos]
    if "container" in names: return "Scene_Container"
    if "wagon" in names:     return "Scene_Wagon"
    return "Scene_Unknown"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--export_json",required=True)
    ap.add_argument("--image_roots",required=True,help="root1;root2;root3 (train/val/test)")
    ap.add_argument("--out_root",required=True,help="Output root (labels/renders/yaml)")
    ap.add_argument("--prefix",default="")
    ap.add_argument("--render",default="true")
    ap.add_argument("--box_thick",type=int,default=6)
    ap.add_argument("--font_scale",type=float,default=0.9)
    ap.add_argument("--font_thick",type=int,default=2)
    args=ap.parse_args()
    do_render=str(args.render).lower() in {"1","true","yes","y"}

    roots=[s for s in args.image_roots.split(";") if s.strip()]
    train_root = next((r for r in roots if "\\train\\" in r.lower() or "/train/" in r.lower()), roots[0] if roots else "")
    val_root   = next((r for r in roots if "\\val\\"   in r.lower() or "/val/"   in r.lower()), "")
    test_root  = next((r for r in roots if "\\test\\"  in r.lower() or "/test/"  in r.lower()), roots[-1] if len(roots)>1 else "")

    print(f"[info] indexing images under: {roots}")
    file_map=list_images_under(*roots)
    print(f"[info] indexed {len(file_map)} images")

    out_root=pathlib.Path(args.out_root)
    for split in ("train","val","test"):
        ensure_dir(out_root/"labels"/split)
        if do_render: ensure_dir(out_root/"renders"/split)

    yaml_p=out_root/"dataset_norm6_scene.yaml"
    with open(yaml_p,"w",encoding="utf-8") as f:
        f.write(
f"""# Auto-generated
path: {out_root.as_posix()}
train: {train_root.replace('\\','/')}
val:   {val_root.replace('\\','/')}
test:  {test_root.replace('\\','/')}
names:
  0: BAW-Sack
  1: Plastic
  2: Metal
  3: Glass
  4: Paper_Cardboard
  5: Other
""")

    tasks=json.load(open(args.export_json,"r",encoding="utf-8"))
    kept=missing=unlabeled=bad=0; scenes_count=defaultdict(int)

    for t in tasks:
        data=t.get("data") or {}; img_url=data.get("image") or data.get("img") or ""
        fname=os.path.basename(img_url).lower()
        if not fname: continue
        img_path=file_map.get(fname)
        if not img_path: missing+=1; continue

        split=split_of_path(img_path,train_root,val_root,test_root)
        stem=(args.prefix or "")+os.path.splitext(os.path.basename(img_path))[0]
        out_lbl=out_root/"labels"/split/f"{stem}.txt"
        out_rnd=out_root/"renders"/split/f"{stem}.jpg"

        im=cv2.imread(img_path); 
        if im is None: missing+=1; continue
        H,W=im.shape[:2]

        annos=parse_labelstudio_annos(t,W,H)
        scene=scene_from_annos(annos); scenes_count[scene]+=1

        lines=[]; local_bad=0
        for a in annos:
            cname=a.get("class_name") or ""
            key=norm(ALIASES.get(cname.strip(),cname))
            if key not in DET_IDX:  # not one of the 6 detection classes
                continue
            box=clip_box(int(a["x1"]),int(a["y1"]),int(a["x2"]),int(a["y2"]),W,H)
            if not box: local_bad+=1; continue
            x1,y1,x2,y2=box; cx,cy,w,h=to_yolo(x1,y1,x2,y2,W,H)
            lines.append(f"{DET_IDX[key]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if lines:
            with open(out_lbl,"w",encoding="utf-8") as f: f.write("\n".join(lines))
            kept+=1
        else:
            unlabeled+=1
        bad+=local_bad

        if do_render:
            canvas=im.copy()
            for a in annos:
                cname=a.get("class_name") or ""
                key=norm(ALIASES.get(cname.strip(),cname))
                if key not in DET_IDX: continue
                box=clip_box(int(a["x1"]),int(a["y1"]),int(a["x2"]),int(a["y2"]),W,H)
                if not box: continue
                x1,y1,x2,y2=box
                cv2.rectangle(canvas,(x1,y1),(x2,y2),(255,0,255),thickness=args.box_thick)
                label=ALIASES.get(cname.strip(),cname)
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, args.font_thick)
                tx=max(x1+2, min(x2-tw-2, x1 + (x2-x1 - tw)//2))
                ty=max(y1+th+2, min(y2-2,  y1 + (y2-y1)//2))
                cv2.rectangle(canvas,(tx-2,ty-th-2),(tx+tw+2,ty+2),(255,0,255),thickness=-1)
                cv2.putText(canvas,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,args.font_scale,(255,255,255),args.font_thick,cv2.LINE_AA)
            text=scene
            (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale*1.2, args.font_thick+1)
            cx=W//2 - tw//2; cy=H//2 + th//2
            cv2.putText(canvas,text,(cx+2,cy+2),cv2.FONT_HERSHEY_SIMPLEX,args.font_scale*1.2,(0,0,0),args.font_thick+2,cv2.LINE_AA)
            cv2.putText(canvas,text,(cx,cy),    cv2.FONT_HERSHEY_SIMPLEX,args.font_scale*1.2,(240,240,240),args.font_thick+1,cv2.LINE_AA)
            cv2.imwrite(str(out_rnd),canvas)

    print("\n[summary]")
    print(f"  Out root            : {out_root}")
    print(f"  Kept label files    : {kept}")
    print(f"  Missing images      : {missing}")
    print(f"  Unlabeled (no det)  : {unlabeled}")
    print(f"  Bad geometry        : {bad}")
    print(f"  YAML                : {yaml_p}")
    print("  Scene counts        :", dict(scenes_count))

if __name__ == "__main__":
    main()
