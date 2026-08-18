#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');a=p.parse_args();c=load(a.config);T=int(c['threshold']);runs=fmt(c['runs_root'],T);target=fmt(c['dataset_output'],T);context=fmt(c['context']['dataset_output'],T);checks={'source_yaml':{'path':c['source_yamls'][0],'exists':Path(c['source_yamls'][0]).is_file()},'target_dataset':{'path':str(target/'detection'),'exists':(target/'detection'/'dataset.yaml').is_file()},'context_dataset':{'path':str(context),'exists':(context/'train'/'container').is_dir()},'existing_D1_yolo11m':{'path':c['context']['existing']['yolo11m'],'exists':Path(c['context']['existing']['yolo11m']).is_file()},'existing_D1_yolov8m':{'path':c['context']['existing']['yolov8m'],'exists':Path(c['context']['existing']['yolov8m']).is_file()},'existing_final_hybrid_A':{'path':c['existing_final_hybrid']['stage_a'],'exists':Path(c['existing_final_hybrid']['stage_a']).is_file()},'existing_final_hybrid_B':{'path':c['existing_final_hybrid']['stage_b'],'exists':Path(c['existing_final_hybrid']['stage_b']).is_file()},'existing_final_hybrid_C':{'path':c['existing_final_hybrid']['stage_c'],'exists':Path(c['existing_final_hybrid']['stage_c']).is_file()},'unified_final_models':{'path':str(runs/'FINAL_MODELS'),'exists':(runs/'FINAL_MODELS'/'final_model_manifest.json').is_file()}}
    print(json.dumps(checks,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
