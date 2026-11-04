#!/usr/bin/env python3
"""Compare base versus UAVVaste-fine-tuned Stage A on TARGET validation only."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def target_view(c,runs):
    local=runs/'TRAINING_ASSETS'/'StageA_Final_Replay_View'
    return local if (local/'dataset.yaml').is_file() else Path(c['existing_phase4b']['stage_a_training_view']).expanduser().resolve()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');a=p.parse_args();c=load(a.config);T=int(c['threshold']);runs=fmt(c['runs_root'],T);base=runs/'BASE_FINAL_MODELS'/'StageA_AnyFM_Detector'/'best.pt';candidate=runs/'UAVVASTE_FINETUNE'/'STAGEA'/'StageA_4B_Finetuned_TargetPlusUAVVaste'/'weights'/'best.pt';data=target_view(c,runs)/'dataset.yaml'
    for x in (base,candidate,data):
        if not x.is_file(): raise FileNotFoundError(x)
    from ultralytics import YOLO
    metrics={}
    for label,path in (('base_4B',base),('uavvaste_finetuned',candidate)):
        r=YOLO(str(path),task='detect').val(data=str(data),split='val',imgsz=int(c['stage_a_uavvaste_finetune']['imgsz']),batch=int(c['stage_a_uavvaste_finetune']['batch']),device=c['device'],conf=0.001,iou=0.70,plots=False,verbose=False)
        metrics[label]={'precision':float(r.box.mp),'recall':float(r.box.mr),'mAP50':float(r.box.map50),'mAP50_95':float(r.box.map)}
    b=metrics['base_4B'];n=metrics['uavvaste_finetuned'];delta={k:n[k]-b[k] for k in b};crit=c['stage_a_acceptance'];checks={'recall_gain':delta['recall']>=float(crit['minimum_recall_gain']),'mAP50_gain':delta['mAP50']>=float(crit['minimum_map50_gain']),'mAP50_95_gain':delta['mAP50_95']>=float(crit['minimum_map50_95_gain']),'precision_drop':delta['precision']>=-float(crit['maximum_precision_drop'])};accepted=all(checks.values());result={'evaluation_domain':'TARGET validation only','stage_a':metrics,'delta':delta,'acceptance_criteria':crit,'checks':checks,'accepted_for_hybrid':accepted,'recommended_stage_a':str(candidate if accepted else base)};out=runs/'UAVVASTE_FINETUNE'/'COMPARISON';out.mkdir(parents=True,exist_ok=True);(out/'stageA_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
