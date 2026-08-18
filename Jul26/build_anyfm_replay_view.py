#!/usr/bin/env python3
"""Build one-class Any-FM views with controlled hard-positive replay and 1:1 exposure."""
from __future__ import annotations
import argparse,csv,json,random,shutil,sys
from pathlib import Path
from phase4_common import *

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--hard-images",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replay-fraction",type=float,required=True); p.add_argument("--negative-ratio",type=float,default=1.0); p.add_argument("--seed",type=int,default=42); p.add_argument("--copy-mode",choices=('hardlink','copy'),default='hardlink'); p.add_argument("--overwrite",action='store_true'); return p.parse_args()

def convert_label(source:Path,target:Path):
    target.parent.mkdir(parents=True,exist_ok=True); lines=[]
    for raw in source.read_text(encoding='utf-8-sig').splitlines():
        tokens=raw.strip().split()
        if tokens: lines.append('0 '+' '.join(tokens[1:]))
    target.write_text(('\n'.join(lines)+'\n') if lines else '',encoding='utf-8')

def main():
    a=parse_args(); out=a.output.expanduser().resolve()
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    root,_,names=read_dataset(a.data); rng=random.Random(a.seed)
    ranked=[]
    with a.hard_images.open('r',encoding='utf-8-sig',newline='') as stream:
        ranked=list(csv.DictReader(stream))
    ranked=[r for r in ranked if float(r['image_hardness'])>0]
    summary={}
    for split in ('train','val'):
        images_root=split_dir(root,'images',split); labels_root=split_dir(root,'labels',split); images=list_images(images_root)
        positives=[]; negatives=[]
        for image in images:
            gt,_,label=image_gt(image,images_root,labels_root,names); (positives if gt else negatives).append(image)
        if split=='train':
            selected_neg=rng.sample(negatives,min(len(negatives),round(len(positives)*a.negative_ratio)))
            selected=positives+selected_neg
        else: selected=images; selected_neg=negatives
        for image in selected:
            rel=image.relative_to(images_root); target_image=out/'images'/split/rel; target_label=out/'labels'/split/rel.with_suffix('.txt')
            transfer(image,target_image,a.copy_mode); convert_label(corresponding_label(image,images_root,labels_root),target_label)
        replay_count=0
        if split=='train' and a.replay_fraction>0:
            requested=round(len(positives)*a.replay_fraction); path_by_rel={p.relative_to(images_root).as_posix():p for p in positives}
            chosen=[]
            for row in ranked:
                path=path_by_rel.get(row['relative_image'])
                if path is not None and path not in chosen: chosen.append(path)
                if len(chosen)>=requested: break
            replay_count=len(chosen)
            for index,image in enumerate(chosen):
                rel=image.relative_to(images_root); name=Path('__replay_pos__')/f'{index:05d}__{rel.name}'
                transfer(image,out/'images'/split/name,a.copy_mode); convert_label(corresponding_label(image,images_root,labels_root),out/'labels'/split/name.with_suffix('.txt'))
            # Equal negative replay preserves exposure balance.
            neg_replay=[rng.choice(selected_neg) for _ in range(replay_count)] if selected_neg else []
            for index,image in enumerate(neg_replay):
                rel=image.relative_to(images_root); name=Path('__replay_neg__')/f'{index:05d}__{rel.name}'
                transfer(image,out/'images'/split/name,a.copy_mode); convert_label(corresponding_label(image,images_root,labels_root),out/'labels'/split/name.with_suffix('.txt'))
        summary[split]={"base_images":len(selected),"base_positive":len(positives) if split=='train' else sum(bool(image_gt(i,images_root,labels_root,names)[0]) for i in images),"base_negative":len(selected_neg),"positive_replay":replay_count,"negative_replay":replay_count if split=='train' else 0,"final_images":len(selected)+2*replay_count}
    write_detection_yaml(out,{0:'ForeignMaterial'})
    (out/'view_summary.json').write_text(json.dumps({"replay_fraction":a.replay_fraction,"negative_ratio":a.negative_ratio,"splits":summary},indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f"ERROR: {exc}",file=sys.stderr); raise SystemExit(1)
