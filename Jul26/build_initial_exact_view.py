#!/usr/bin/env python3
"""Create split-preserving four-class GT crop data for the initial exact classifier."""
from __future__ import annotations
import argparse, json, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from phase4_common import read_dataset, split_dir, list_images, image_gt, crop_box, transfer

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--crop-margin',type=float,default=0.15); p.add_argument('--balance-target',type=float,default=0.90); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--copy-mode',choices=('hardlink','copy'),default='hardlink'); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def main():
    a=parse_args(); out=a.output.expanduser().resolve()
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    root,_,names=read_dataset(a.data); rng=random.Random(a.seed); summary={}
    for split in ('train','val'):
        images_root=split_dir(root,'images',split); labels_root=split_dir(root,'labels',split); rows=defaultdict(list)
        for image in list_images(images_root):
            gt,size,_=image_gt(image,images_root,labels_root,names)
            if not gt: continue
            with Image.open(image) as source: rgb=source.convert('RGB')
            for index,item in enumerate(gt):
                left,top,right,bottom=crop_box(item['bbox'],size,a.crop_margin)
                if right-left<8 or bottom-top<8: continue
                cls=names[int(item['class_id'])]; target=out/split/cls/f'{image.stem}__g{index:04d}.jpg'; target.parent.mkdir(parents=True,exist_ok=True)
                rgb.crop((left,top,right,bottom)).save(target,quality=94); rows[cls].append(target)
        original={k:len(v) for k,v in rows.items()}; replay=Counter()
        if split=='train' and rows:
            target_count=round(max(original.values())*a.balance_target)
            for cls,paths in rows.items():
                for idx in range(max(0,target_count-len(paths))):
                    source=rng.choice(paths); target=out/split/cls/f'__balance__{idx:05d}__{source.name}'; transfer(source,target,a.copy_mode); replay[cls]+=1
        summary[split]={'original_counts':original,'balance_replay':dict(replay),'final_counts':{cls:len(list((out/split/cls).glob("*"))) for cls in rows}}
    (out/'view_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
