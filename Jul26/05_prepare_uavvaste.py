#!/usr/bin/env python3
"""Download/extract official UAVVaste and convert COCO bbox annotations to one-class YOLO."""
from __future__ import annotations
import argparse, csv, json, shutil, sys, zipfile
from collections import defaultdict
from pathlib import Path
import requests, yaml
from common import load_config, md5, list_images, transfer

ROOT=Path(__file__).resolve().parent

def locate(root,name):
    found=list(Path(root).rglob(name))
    if not found:
        raise FileNotFoundError(f"{name} not found under {root}")
    return found[0]

def download(url,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(target.suffix+".part")
    with requests.get(url,stream=True,timeout=(30,300)) as response:
        response.raise_for_status()
        total=int(response.headers.get("content-length",0))
        done=0
        with temp.open("wb") as stream:
            for chunk in response.iter_content(8*1024*1024):
                if not chunk: continue
                stream.write(chunk); done+=len(chunk)
                if total:
                    print(
                        f"\rDownloading UAVVaste: {100*done/total:6.2f}% "
                        f"({done/1e9:.2f}/{total/1e9:.2f} GB)",
                        end="",flush=True
                    )
        print()
    temp.replace(target)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=ROOT/"config.json")
    p.add_argument("--download",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    c=load_config(a.config)
    u=c["uavvaste"]
    workspace=Path(u["workspace"]).expanduser().resolve()
    archive=Path(u["archive"]).expanduser().resolve()
    prepared=Path(u["prepared_yolo"]).expanduser().resolve()
    extracted=workspace/"extracted"

    if a.download and not archive.is_file():
        download(u["official_url"],archive)
    if not archive.is_file() and not extracted.is_dir():
        raise FileNotFoundError(
            f"UAVVaste ZIP not found: {archive}\n"
            "Use --download or place the official ZIP there."
        )
    if archive.is_file():
        actual=md5(archive)
        expected=u["expected_md5"]
        if actual.casefold()!=expected.casefold():
            raise RuntimeError(
                f"UAVVaste ZIP MD5 mismatch: {actual}; expected {expected}"
            )
        if not extracted.is_dir():
            extracted.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extracted)

    annotations=json.loads(locate(extracted,"annotations.json").read_text(encoding="utf-8-sig"))
    distribution=json.loads(
        locate(extracted,"train_val_test_distribution_file.json").read_text(encoding="utf-8-sig")
    )
    expected=u["expected_split_counts"]
    actual={k:len(distribution.get(k,[])) for k in expected}
    if actual!=expected:
        raise RuntimeError(f"Unexpected UAVVaste splits: {actual}; expected {expected}")

    categories=annotations.get("categories",[])
    if len(categories)!=1 or str(categories[0]["name"]).casefold()!="rubbish":
        raise RuntimeError(f"Unexpected UAVVaste taxonomy: {categories}")

    if prepared.exists():
        if not a.overwrite:
            print(f"Prepared UAVVaste already exists: {prepared}")
            return 0
        shutil.rmtree(prepared)

    image_records={
        str(item["file_name"]).casefold():item
        for item in annotations["images"]
    }
    anns=defaultdict(list)
    for ann in annotations["annotations"]:
        anns[int(ann["image_id"])].append(ann)

    source_index={
        p.name.casefold():p
        for p in list_images(extracted)
    }

    manifest=[]
    stats={}
    for split,names in distribution.items():
        if split not in expected: continue
        images=positive=objects=0
        for filename in names:
            record=image_records[filename.casefold()]
            source=source_index.get(filename.casefold())
            if source is None:
                raise FileNotFoundError(filename)
            W,H=int(record["width"]),int(record["height"])
            dst=prepared/"images"/split/filename
            transfer(source,dst,"hardlink")
            label=prepared/"labels"/split/Path(filename).with_suffix(".txt")
            poly=prepared/"polygons"/split/Path(filename).with_suffix(".txt")
            label.parent.mkdir(parents=True,exist_ok=True)
            poly.parent.mkdir(parents=True,exist_ok=True)
            lines=[]; polys=[]
            for ann in anns[int(record["id"])]:
                x,y,bw,bh=map(float,ann["bbox"])
                x1=max(0.0,min(W,x)); y1=max(0.0,min(H,y))
                x2=max(0.0,min(W,x+bw)); y2=max(0.0,min(H,y+bh))
                if x2<=x1 or y2<=y1: continue
                xc=((x1+x2)/2)/W; yc=((y1+y2)/2)/H
                nw=(x2-x1)/W; nh=(y2-y1)/H
                lines.append(f"0 {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}\n")
                seg=ann.get("segmentation",[])
                if isinstance(seg,list):
                    for polygon in seg:
                        if not isinstance(polygon,list) or len(polygon)<6 or len(polygon)%2:
                            continue
                        values=[]
                        for px,py in zip(polygon[0::2],polygon[1::2]):
                            values.extend((
                                max(0,min(1,float(px)/W)),
                                max(0,min(1,float(py)/H))
                            ))
                        polys.append("0 "+" ".join(f"{v:.8f}" for v in values)+"\n")
            label.write_text("".join(lines),encoding="utf-8")
            poly.write_text("".join(polys),encoding="utf-8")
            images+=1; positive+=int(bool(lines)); objects+=len(lines)
            manifest.append({
                "split":split,"source":"UAVVaste","image":str(dst),
                "label":str(label),"objects":len(lines)
            })
        stats[split]={"images":images,"positive_images":positive,"objects":objects}

    (prepared/"dataset.yaml").write_text(
        yaml.safe_dump({
            "path":str(prepared),
            "train":"images/train","val":"images/val","test":"images/test",
            "names":{0:"ForeignMaterial"}
        },sort_keys=False),
        encoding="utf-8"
    )
    with (prepared/"provenance_manifest.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    (prepared/"conversion_summary.json").write_text(
        json.dumps({
            "mapping":{"rubbish":"ForeignMaterial"},
            "statistics":stats,
            "policy":"Only official UAVVaste train is later used for fine-tuning."
        },indent=2),
        encoding="utf-8"
    )
    print(json.dumps(stats,indent=2))
    return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr); raise SystemExit(1)
