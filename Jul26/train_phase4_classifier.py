#!/usr/bin/env python3
"""Generic Phase-4 classifier trainer for binary verifier or four-class exact classifier."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--project',type=Path,required=True); p.add_argument('--name',required=True); p.add_argument('--model',required=True); p.add_argument('--role',choices=('verifier','exact'),required=True); p.add_argument('--imgsz',type=int,required=True); p.add_argument('--epochs',type=int,required=True); p.add_argument('--patience',type=int,required=True); p.add_argument('--batch',type=int,required=True); p.add_argument('--device',default='0'); p.add_argument('--workers',type=int,default=8); p.add_argument('--seed',type=int,default=42); p.add_argument('--optimizer',default='AdamW'); p.add_argument('--lr0',type=float,required=True); p.add_argument('--lrf',type=float,default=0.01); p.add_argument('--weight-decay',type=float,default=0.0005); p.add_argument('--warmup-epochs',type=float,default=3.0); p.add_argument('--dropout',type=float,default=0.05); p.add_argument('--overwrite',action='store_true'); p.add_argument('--amp',choices=('true','false'),default='true'); p.add_argument('--skip-amp-check',action='store_true'); return p.parse_args()
def main():
    a=parse_args(); run=a.project.expanduser().resolve()/a.name
    if run.exists():
        if not a.overwrite: raise FileExistsError(run)
        shutil.rmtree(run)
    from ultralytics import YOLO
    from ultralytics_runtime import suppress_auxiliary_amp_model_check
    suppress_auxiliary_amp_model_check(a.skip_amp_check)
    model=YOLO(a.model,task='classify')
    args={"data":str(a.data.expanduser().resolve()),"epochs":a.epochs,"patience":a.patience,"imgsz":a.imgsz,"batch":a.batch,"device":a.device,"workers":a.workers,"project":str(a.project.expanduser().resolve()),"name":a.name,"exist_ok":True,"seed":a.seed,"deterministic":True,"optimizer":a.optimizer,"lr0":a.lr0,"lrf":a.lrf,"weight_decay":a.weight_decay,"warmup_epochs":a.warmup_epochs,"dropout":a.dropout,"pretrained":True,"amp":a.amp=='true',"plots":True,"val":True,"degrees":5.0,"translate":0.05,"scale":0.15,"fliplr":0.50,"flipud":0.0,"hsv_h":0.015,"hsv_s":0.20,"hsv_v":0.15,"erasing":0.10,"auto_augment":"randaugment"}
    a.project.mkdir(parents=True,exist_ok=True); results=model.train(**args); save=Path(getattr(results,'save_dir',run)); (save/'requested_config.json').write_text(json.dumps({'role':a.role,'training_arguments':args},indent=2),encoding='utf-8'); print(f'Best checkpoint: {save/"weights"/"best.pt"}'); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
