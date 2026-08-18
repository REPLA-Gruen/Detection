#!/usr/bin/env python3
"""Build controlled binary-verifier and exact-classifier views from the crop pool."""
from __future__ import annotations
import argparse,csv,json,random,shutil,sys
from collections import Counter
from pathlib import Path
from phase4_common import transfer

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--pool',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--background-ratios',type=float,nargs='+',required=True); p.add_argument('--hard-replay-fractions',type=float,nargs='+',required=True); p.add_argument('--negative-image-fraction',type=float,default=0.75); p.add_argument('--seed',type=int,default=42); p.add_argument('--copy-mode',choices=('hardlink','copy'),default='hardlink'); p.add_argument('--overwrite',action='store_true'); return p.parse_args()
def tag(v): return f'{v:.2f}'.replace('.','p')
def main():
    a=parse_args(); out=a.output.expanduser().resolve()
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    with (a.pool/'crop_pool_manifest.csv').open('r',encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
    rng=random.Random(a.seed); summary={'binary':{},'exact':{}}
    # Binary views: copy all FO crops, choose hard backgrounds by detector confidence.
    for ratio in a.background_ratios:
        root=out/'binary_views'/f'BG_{tag(ratio)}'; split_summary={}
        for split in ('train','val'):
            fo=[r for r in rows if r['split']==split and r['kind']=='foreign_material']; bg=[r for r in rows if r['split']==split and r['kind']=='background']
            for idx,r in enumerate(fo): transfer(Path(r['crop_path']),root/split/'ForeignMaterial'/f'{idx:06d}__{Path(r["crop_path"]).name}',a.copy_mode)
            if split=='train':
                target=round(len(fo)*ratio); neg=sorted([r for r in bg if r['source_group']=='negative_image'],key=lambda r:float(r['detector_confidence']),reverse=True); pos=sorted([r for r in bg if r['source_group']=='positive_image_background'],key=lambda r:float(r['detector_confidence']),reverse=True); wanted=round(target*a.negative_image_fraction); selected=neg[:wanted]; selected+=pos[:max(0,target-len(selected))];
                if len(selected)<target:
                    used={r['crop_path'] for r in selected}; fallback=sorted([r for r in bg if r['crop_path'] not in used],key=lambda r:float(r['detector_confidence']),reverse=True); selected+=fallback[:target-len(selected)]
            else: selected=bg
            for idx,r in enumerate(selected): transfer(Path(r['crop_path']),root/split/'Background_NotFO'/f'{idx:06d}__{Path(r["crop_path"]).name}',a.copy_mode)
            split_summary[split]={'foreign_material':len(fo),'background':len(selected)}
        (root/'view_summary.json').write_text(json.dumps({'background_ratio':ratio,'splits':split_summary},indent=2),encoding='utf-8'); summary['binary'][root.name]=split_summary
    # Exact views: class-balance toward 90% of max support, then replay hard crops.
    for fraction in a.hard_replay_fractions:
        root=out/'exact_views'/f'Hard_{tag(fraction)}'; split_summary={}
        for split in ('train','val'):
            fo=[r for r in rows if r['split']==split and r['kind']=='foreign_material']; by_class={}
            for r in fo: by_class.setdefault(r['class_name'],[]).append(r)
            original_counts={k:len(v) for k,v in by_class.items()}; selected=list(fo); replay=[]
            if split=='train':
                target=round(max(original_counts.values())*0.90)
                for cls,items in by_class.items():
                    need=max(0,target-len(items)); hard=[r for r in items if str(r['exact_hard']).casefold()=='true']; hard.sort(key=lambda r:(float(r['exact_margin']),float(r['exact_confidence'])))
                    source=hard or items
                    replay.extend([rng.choice(source) for _ in range(need)])
                cap=round(len(fo)*fraction); hard_all=[r for r in fo if str(r['exact_hard']).casefold()=='true']; hard_all.sort(key=lambda r:(float(r['exact_margin']),float(r['exact_confidence'])))
                for r in hard_all:
                    if len(replay)>=cap: break
                    replay.append(r)
                replay=replay[:cap]
            all_rows=selected+replay
            for idx,r in enumerate(all_rows): transfer(Path(r['exact_crop_path']),root/split/r['class_name']/f'{idx:06d}__{Path(r["exact_crop_path"]).name}',a.copy_mode)
            split_summary[split]={'original_counts':original_counts,'replay':len(replay),'final_total':len(all_rows)}
        (root/'view_summary.json').write_text(json.dumps({'hard_replay_fraction':fraction,'splits':split_summary},indent=2),encoding='utf-8'); summary['exact'][root.name]=split_summary
    (out/'all_views_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
