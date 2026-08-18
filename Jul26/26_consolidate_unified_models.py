#!/usr/bin/env python3
"""Create ONE deployable FINAL_MODELS containing Stage 0 + Stages A/B/C."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for ch in iter(lambda:f.read(1024*1024),b''):h.update(ch)
    return h.hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);return dst
def resolve_context(runs):
    decision=runs/'CONTEXT_SELECTION'/'selected_context_model.json'
    if decision.is_file():
        d=load(decision);return Path(d['selected_checkpoint']),d
    imported=runs/'CONTEXT_IMPORTED_MODELS'/'YOLO11m'/'best.pt'
    if imported.is_file():return imported,{'selected_family':'yolo11m','selected_checkpoint':str(imported),'accepted':True,'selection_basis':'legacy import default; run evaluate-context for explicit aligned-validation evidence'}
    raise FileNotFoundError('No selected context model. Run evaluate-context or import-context.')
def resolve_foreign(c,runs):
    imported=runs/'FOREIGN_IMPORTED_FINAL_MODELS'
    if all((imported/folder/'best.pt').is_file() for folder in ('StageA_AnyFM_Detector','StageB_Binary_Verifier','StageC_Exact_Classifier')):
        return {'stage_a':imported/'StageA_AnyFM_Detector'/'best.pt','stage_b':imported/'StageB_Binary_Verifier'/'best.pt','stage_c':imported/'StageC_Exact_Classifier'/'best.pt'},{'mode':'imported current final hybrid','source':str(imported)}
    base=runs/'BASE_FINAL_MODELS';comparison=runs/'UAVVASTE_FINETUNE'/'COMPARISON'/'stageA_comparison.json';candidate=runs/'UAVVASTE_FINETUNE'/'STAGEA'/'StageA_4B_Finetuned_TargetPlusUAVVaste'/'weights'/'best.pt'
    if not comparison.is_file():raise FileNotFoundError('No imported final hybrid and no Stage-A comparison. Run import-foreign or train/finetune/compare.')
    d=load(comparison);a=candidate if bool(d.get('accepted_for_hybrid')) else base/'StageA_AnyFM_Detector'/'best.pt';models={'stage_a':a,'stage_b':base/'StageB_Binary_Verifier'/'best.pt','stage_c':base/'StageC_Exact_Classifier'/'best.pt'}
    return models,{'mode':'composed from unified training','stage_a_comparison':d}
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');p.add_argument('--allow-context-below-floor',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);runs=fmt(c['runs_root'],T);context,context_decision=resolve_context(runs)
    if not bool(context_decision.get('accepted',True)) and not a.allow_context_below_floor:raise RuntimeError('Context selection did not meet validation floor; use --allow-context-below-floor only for engineering.')
    foreign,foreign_prov=resolve_foreign(c,runs);sources={'context':context,**foreign}
    for pth in sources.values():
        if not Path(pth).is_file():raise FileNotFoundError(pth)
    final=runs/'FINAL_MODELS'
    if final.exists():
        if not a.overwrite:raise FileExistsError(f'{final} exists; add --overwrite')
        shutil.rmtree(final)
    destinations={'context':final/'Stage0_Context_Classifier'/'best.pt','stage_a':final/'StageA_AnyFM_Detector'/'best.pt','stage_b':final/'StageB_Binary_Verifier'/'best.pt','stage_c':final/'StageC_Exact_Classifier'/'best.pt'}
    for key,src in sources.items():copy(Path(src),destinations[key])
    # Include YOLOv8m D1 reference when available, but it is not loaded by default.
    ref=None
    for cand in [runs/'CONTEXT_TRAINING_RUNS'/c['context']['training']['yolov8_run_name']/'weights'/'best.pt',runs/'CONTEXT_IMPORTED_MODELS'/'YOLOv8m'/'best.pt']:
        if cand.is_file():ref=final/'Stage0_Context_Reference_YOLOv8m'/'best.pt';copy(cand,ref);break
    manifest={'product':'CCBRE Context + Hybrid Foreign-Material Final Product','version':'1.0','architecture':['Stage0_Context_Classifier','StageA_AnyFM_Detector','StageB_Binary_Verifier','StageC_Exact_Classifier'],'context_role':'whole-image Container/Wagon metadata; it does not suppress foreign-material inference in this final pre-Phase-5 product','models':{k:str(v.resolve()) for k,v in destinations.items()},'optional_context_reference_yolov8m':str(ref.resolve()) if ref else None,'context_selection':context_decision,'foreign_provenance':foreign_prov,'default_operating_point':c['default_operating_point'],'operating_points':c['operating_points'],'sha256':{k:sha(v) for k,v in destinations.items()}}
    final.mkdir(parents=True,exist_ok=True);(final/'final_model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
