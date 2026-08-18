#!/usr/bin/env python3
"""Copy the user's existing final 4B A/B/C models into this product's BASE_FINAL_MODELS."""
from __future__ import annotations
import argparse, json, shutil, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T): return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def sha(path):
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'config.json'); p.add_argument('--overwrite',action='store_true'); a=p.parse_args(); c=load(a.config); T=int(c['threshold'])
    runs=fmt(c['runs_root'],T); target=runs/'BASE_FINAL_MODELS'; src=c['existing_phase4b']
    mapping={'stage_a':(Path(src['stage_a']),target/'StageA_AnyFM_Detector'/'best.pt'), 'stage_b':(Path(src['stage_b']),target/'StageB_Binary_Verifier'/'best.pt'), 'stage_c':(Path(src['stage_c']),target/'StageC_Exact_Classifier'/'best.pt')}
    if target.exists() and a.overwrite: shutil.rmtree(target)
    target.mkdir(parents=True,exist_ok=True); models={}
    for key,(source,dest) in mapping.items():
        source=source.expanduser().resolve()
        if not source.is_file(): raise FileNotFoundError(source)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.is_file() or a.overwrite: shutil.copy2(source,dest)
        if sha(source)!=sha(dest): raise RuntimeError(f'Copy verification failed: {key}')
        models[key]=str(dest.resolve())
    manifest={'source':'existing final 4B models','source_root':src['models_root'],'models':models,'sha256':{k:sha(v) for k,v in [(k,Path(p)) for k,p in models.items()]},'stage_a_training_view':src['stage_a_training_view']}
    (target/'base_model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
