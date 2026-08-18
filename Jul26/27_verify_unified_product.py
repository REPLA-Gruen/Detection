#!/usr/bin/env python3
"""Verify model existence, hashes, roles, and optional model loading for the four-stage product."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for ch in iter(lambda:f.read(1024*1024),b''):h.update(ch)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--load-models',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);final=fmt(c['runs_root'],T)/'FINAL_MODELS';m=load(final/'final_model_manifest.json');checks={};paths={k:Path(v) for k,v in m['models'].items()}
    expected={'context':'Stage0_Context_Classifier','stage_a':'StageA_AnyFM_Detector','stage_b':'StageB_Binary_Verifier','stage_c':'StageC_Exact_Classifier'}
    for key,path in paths.items():checks[f'{key}_exists']=path.is_file();checks[f'{key}_hash_matches_manifest']=path.is_file() and m.get('sha256',{}).get(key)==sha(path);checks[f'{key}_folder_role']=expected[key] in path.parts
    checks['context_selection_present']=bool(m.get('context_selection'))
    checks['foreign_provenance_present']=bool(m.get('foreign_provenance'))
    loads={}
    if a.load_models:
        from ultralytics import YOLO
        for key,task in [('context','classify'),('stage_a','detect'),('stage_b','classify'),('stage_c','classify')]:
            try:YOLO(str(paths[key]),task=task);loads[key]='PASS'
            except Exception as exc:loads[key]=f'FAIL: {exc}'
    result={'final_models_root':str(final),'checks':checks,'optional_model_load':loads,'passed':all(checks.values()) and all(v=='PASS' for v in loads.values())};(final/'FINAL_PRODUCT_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return 0 if result['passed'] else 2
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
