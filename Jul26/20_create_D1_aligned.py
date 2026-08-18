#!/usr/bin/env python3
"""Create D1 Container/Wagon classification data aligned to the T-Core master split.

This avoids creating an independent random D1 split. Every image keeps the same
train/val/test membership as T-Core+Other, so the unified Stage-0 and A/B/C
components refer to the same master image partitions.
"""
from __future__ import annotations
import argparse,csv,json,os,shutil,sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VALID={'container','wagon'}

def load_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def transfer(src,dst,mode):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():dst.unlink()
    if mode=='hardlink':
        try:os.link(src,dst);return
        except OSError:pass
    shutil.copy2(src,dst)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load_json(a.config);T=int(c['threshold'])
    manifest=fmt(c['context']['source_manifest'],T);output=fmt(c['context']['dataset_output'],T)
    if not manifest.is_file():raise FileNotFoundError(f'T-Core image_manifest.csv not found: {manifest}. Run create-target first.')
    if output.exists():
        if not a.overwrite:raise FileExistsError(f'{output} exists; add --overwrite')
        shutil.rmtree(output)
    output.mkdir(parents=True)
    rows=[];warnings=[];seen={};counts=defaultdict(Counter)
    with manifest.open('r',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            split=str(row.get('split','')).strip().casefold();context=str(row.get('context','')).strip().casefold();digest=str(row.get('sha256','')).strip()
            if split not in {'train','val','test'}:continue
            if context not in VALID:
                warnings.append(f"SKIPPED unknown context: {row.get('source_image','')}")
                continue
            prior=seen.get(digest)
            if prior is not None and prior!=split:raise RuntimeError(f'Exact duplicate hash crosses T-Core splits: {digest}: {prior} vs {split}')
            seen[digest]=split
            src=Path(row.get('output_image') or row.get('source_image','')).expanduser().resolve()
            if not src.is_file():
                src=Path(row.get('source_image','')).expanduser().resolve()
            if not src.is_file():raise FileNotFoundError(src)
            dst=output/split/context/src.name
            transfer(src,dst,c.get('copy_mode','hardlink'))
            counts[split][context]+=1
            rows.append({'split':split,'class':context,'sha256':digest,'source_image':str(src),'output_image':str(dst),'tcore_manifest':str(manifest)})
    reports=output/'reports';reports.mkdir(parents=True,exist_ok=True)
    fields=['split','class','sha256','source_image','output_image','tcore_manifest']
    with (reports/'image_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    dist=[]
    for split in ('train','val','test'):
        dist.append({'split':split,'container':counts[split]['container'],'wagon':counts[split]['wagon'],'total':sum(counts[split].values())})
    with (reports/'distribution_by_split.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['split','container','wagon','total']);w.writeheader();w.writerows(dist)
    (reports/'warnings.txt').write_text('\n'.join(warnings)+('\n' if warnings else ''),encoding='utf-8')
    summary={'task':'D1 context classification','split_policy':'exactly aligned to T-Core image_manifest.csv','source_manifest':str(manifest),'output':str(output),'images':len(rows),'counts':dist,'skipped_unknown_context':len(warnings),'classes':['container','wagon']}
    (reports/'build_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
