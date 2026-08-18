#!/usr/bin/env python3
"""Shared Phase-4 data and geometry helpers."""
from __future__ import annotations
import hashlib, os, shutil
from pathlib import Path
from typing import Any
import yaml
from PIL import Image
from tcore_train_common import (
    corresponding_label, list_images, names_to_dict, resolve_detection_dataset,
    split_dir, validate_names,
)
from evaluate_TCoreOther_comprehensive import parse_ground_truth

IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def normalize_name(value:str)->str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())

def transfer(source:Path,target:Path,mode:str="hardlink")->None:
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): target.unlink()
    if mode=="hardlink":
        try:
            os.link(source,target); return
        except OSError: pass
    shutil.copy2(source,target)

def sha256_file(path:Path,chunk:int=1024*1024)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block=stream.read(chunk)
            if not block: break
            h.update(block)
    return h.hexdigest()

def read_dataset(data:Path):
    root,yaml_path=resolve_detection_dataset(data)
    raw=yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    names=names_to_dict(raw.get("names")); validate_names(names)
    return root,yaml_path,names

def image_gt(image:Path,images_root:Path,labels_root:Path,names:dict[int,str]):
    label=corresponding_label(image,images_root,labels_root)
    with Image.open(image) as source: size=source.size
    return parse_ground_truth(label,size,names),size,label

def context_from_filename(path:Path)->str:
    stem=path.stem.casefold()
    if stem.endswith("w"): return "Wagon"
    if stem.endswith(("c","b","t")): return "Container"
    return "Unknown"

def size_bin(box:list[float],image_size:tuple[int,int])->str:
    width,height=image_size
    area=max(0.0,box[2]-box[0])*max(0.0,box[3]-box[1])
    ratio=area/max(1.0,float(width*height))
    if ratio<0.01: return "small"
    if ratio<0.05: return "medium"
    return "large"

def crop_box(box:list[float],image_size:tuple[int,int],margin:float)->tuple[int,int,int,int]:
    width,height=image_size; x1,y1,x2,y2=box
    mx=(x2-x1)*margin; my=(y2-y1)*margin
    left=max(0,int(x1-mx)); top=max(0,int(y1-my))
    right=min(width,int(x2+mx+0.999)); bottom=min(height,int(y2+my+0.999))
    return left,top,right,bottom

def write_detection_yaml(root:Path,names:dict[int,str],include_test:bool=False)->Path:
    payload={"path":str(root),"train":"images/train","val":"images/val","names":names,"nc":len(names)}
    if include_test and (root/"images"/"test").is_dir(): payload["test"]="images/test"
    path=root/"dataset.yaml"; path.write_text(yaml.safe_dump(payload,sort_keys=False,allow_unicode=True),encoding="utf-8")
    return path
