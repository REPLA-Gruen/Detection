#!/usr/bin/env python3
"""Batch inference for a folder using Stage 0 + A/B/C; writes CSV and per-image JSON."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from context_inference import ContextClassifier
from phase4_inference import Pipeline
ROOT=Path(__file__).resolve().parent
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--recursive',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);final=fmt(c['runs_root'],T)/'FINAL_MODELS';m=load(final/'final_model_manifest.json');op=c['default_operating_point'];settings=c['operating_points'][op];context=ContextClassifier(m['models']['context'],c['device'],c['context']['app']['imgsz']);pipe=Pipeline(m['models']['stage_a'],m['models']['stage_b'],m['models']['stage_c'],settings,c['device']);root=a.input.resolve();files=sorted((root.rglob('*') if a.recursive else root.glob('*')));files=[x for x in files if x.is_file() and x.suffix.casefold() in EXTS];a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for i,image in enumerate(files,1):
        ctx=context.analyze(image);_,fm=pipe.analyze(image);record={'image':str(image),'context':ctx,'foreign_material':fm};j=a.output/f'{image.stem}__analysis.json';j.write_text(json.dumps(record,indent=2),encoding='utf-8');rows.append({'image':str(image),'context':ctx['class_name'],'context_confidence':ctx['confidence'],'foreign_material_present':fm['foreign_material_present'],'verified_objects':len(fm['detections']),'json':str(j)});print(f'[{i}/{len(files)}] {image.name}: context={ctx["class_name"]} FO={fm["foreign_material_present"]}')
    with (a.output/'batch_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['image']);w.writeheader();w.writerows(rows)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
