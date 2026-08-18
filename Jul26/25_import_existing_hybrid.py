#!/usr/bin/env python3
"""Import the current final Hybrid A/B/C model set into the unified product workspace."""
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
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);target=fmt(c['runs_root'],T)/'FOREIGN_IMPORTED_FINAL_MODELS';src=c['existing_final_hybrid']
    if target.exists() and a.overwrite:shutil.rmtree(target)
    mapping={'stage_a':(Path(src['stage_a']),target/'StageA_AnyFM_Detector'/'best.pt'),'stage_b':(Path(src['stage_b']),target/'StageB_Binary_Verifier'/'best.pt'),'stage_c':(Path(src['stage_c']),target/'StageC_Exact_Classifier'/'best.pt')};models={};hashes={}
    for key,(source,dst) in mapping.items():
        source=source.expanduser().resolve();
        if not source.is_file():raise FileNotFoundError(source)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
        if sha(source)!=sha(dst):raise RuntimeError(f'Copy verification failed: {key}')
        models[key]=str(dst.resolve());hashes[key]=sha(dst)
    manifest={'product':'current Hybrid 4B + UAVVaste Stage-A','source_root':src['models_root'],'models':models,'sha256':hashes,'source_files_untouched':True}
    (target/'import_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
