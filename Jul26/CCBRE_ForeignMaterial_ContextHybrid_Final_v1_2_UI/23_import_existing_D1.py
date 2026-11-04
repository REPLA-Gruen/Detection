#!/usr/bin/env python3
"""Import existing D1 checkpoints without modifying the source models."""
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
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);root=fmt(c['runs_root'],T)/'CONTEXT_IMPORTED_MODELS';src=c['context']['existing']
    if root.exists() and a.overwrite:shutil.rmtree(root)
    models={}
    for family,key,folder in [('yolo11m','yolo11m','YOLO11m'),('yolov8m','yolov8m','YOLOv8m')]:
        source=Path(src[key]).expanduser().resolve()
        if not source.is_file():
            print(f'WARNING: existing {family} not found: {source}');continue
        dst=root/folder/'best.pt';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
        if sha(source)!=sha(dst):raise RuntimeError(f'Hash verification failed: {family}')
        models[family]=str(dst.resolve())
    if not models:raise FileNotFoundError('No existing D1 checkpoint was found.')
    manifest={'provenance':'legacy D1 checkpoints imported; source files untouched','models':models,'scientific_note':'An evaluation is independent only if its split was not used to train/select the imported checkpoint. The old D1 report did not establish an isolated true-test result.'}
    root.mkdir(parents=True,exist_ok=True);(root/'import_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
