#!/usr/bin/env python3
"""Verify aligned D1 folders, class counts, image readability, and split hash isolation."""
from __future__ import annotations
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for ch in iter(lambda:f.read(1024*1024),b''):h.update(ch)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');a=p.parse_args();c=load(a.config);T=int(c['threshold']);root=fmt(c['context']['dataset_output'],T)
    checks={};counts={};by_hash=defaultdict(set);errors=[]
    for split in ('train','val','test'):
        counts[split]={}
        for cls in ('container','wagon'):
            folder=root/split/cls
            ims=sorted(x for x in folder.rglob('*') if x.is_file() and x.suffix.casefold() in EXTS) if folder.is_dir() else []
            counts[split][cls]=len(ims)
            if not ims:errors.append(f'Missing/empty: {folder}')
            for image in ims:
                try:
                    with Image.open(image) as im: im.verify()
                except Exception as exc:errors.append(f'Unreadable image {image}: {exc}');continue
                by_hash[sha(image)].add(split)
    cross={h:sorted(v) for h,v in by_hash.items() if len(v)>1}
    if cross:errors.append(f'{len(cross)} exact image hashes cross D1 splits')
    total=sum(sum(v.values()) for v in counts.values())
    result={'dataset':str(root),'counts':counts,'total_images':total,'cross_split_exact_duplicate_hashes':len(cross),'errors':errors,'passed':not errors}
    reports=root/'verification';reports.mkdir(parents=True,exist_ok=True);(reports/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return 0 if result['passed'] else 2
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
