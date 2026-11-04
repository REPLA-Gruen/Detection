#!/usr/bin/env python3
"""Verify converted UAVVaste and render random GT samples."""
from __future__ import annotations
import argparse, json, random, shutil, sys
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def font(size):
    for name in (r'C:\Windows\Fonts\arial.ttf','arial.ttf','DejaVuSans.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'config.json'); p.add_argument('--render-count',type=int,default=20); p.add_argument('--overwrite',action='store_true'); a=p.parse_args(); c=load(a.config); root=Path(c['uavvaste']['prepared_yolo']).expanduser().resolve()
    raw=yaml.safe_load((root/'dataset.yaml').read_text(encoding='utf-8-sig')); names=raw['names']; n0=names[0] if isinstance(names,list) else names.get(0,names.get('0'))
    if n0!='ForeignMaterial': raise RuntimeError(f'Unexpected taxonomy: {names}')
    out=root/'verification';
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True); rng=random.Random(c['seed']); summary={}
    expected=c['uavvaste']['expected_split_counts']
    for split in ('train','val','test'):
        images=sorted(p for p in (root/'images'/split).rglob('*') if p.is_file() and p.suffix.casefold() in EXTS)
        if len(images)!=int(expected[split]): raise RuntimeError(f'{split}: {len(images)} != expected {expected[split]}')
        counts=Counter()
        for image in images:
            label=root/'labels'/split/image.with_suffix('.txt').name
            if not label.is_file(): raise FileNotFoundError(label)
            lines=[x for x in label.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
            for no,line in enumerate(lines,1):
                parts=line.split();
                if len(parts)!=5 or int(float(parts[0]))!=0: raise ValueError(f'{label}:{no}')
                vals=list(map(float,parts[1:]));
                if not all(0<=v<=1 for v in vals) or vals[2]<=0 or vals[3]<=0: raise ValueError(f'{label}:{no}')
            counts['images']+=1; counts['positive_images']+=int(bool(lines)); counts['negative_images']+=int(not lines); counts['objects']+=len(lines)
        summary[split]=dict(counts)
        samples=rng.sample(images,min(a.render_count,len(images))); render_dir=out/'rendered'/split; render_dir.mkdir(parents=True,exist_ok=True)
        for image_path in samples:
            with Image.open(image_path) as src: im=src.convert('RGB')
            d=ImageDraw.Draw(im); f=font(max(14,round(min(im.size)/55))); W,H=im.size
            label=root/'labels'/split/image_path.with_suffix('.txt').name
            for i,line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(),1):
                if not line.strip():continue
                _,xc,yc,bw,bh=line.split(); xc,yc,bw,bh=map(float,(xc,yc,bw,bh)); x1=(xc-bw/2)*W;y1=(yc-bh/2)*H;x2=(xc+bw/2)*W;y2=(yc+bh/2)*H
                d.rectangle((x1,y1,x2,y2),outline='red',width=max(2,round(min(W,H)/400))); d.text((x1,max(0,y1-20)),f'FO{i}',fill='red',font=f)
            im.save(render_dir/f'{image_path.stem}__GT.jpg',quality=92)
    (out/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
