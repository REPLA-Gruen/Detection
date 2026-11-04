#!/usr/bin/env python3
"""Build binary-verifier and exact-classifier development datasets.

Positive crops are GT boxes. Background crops are high-confidence detector proposals
with negligible overlap with every GT object. Existing four-class classifier scores
identify hard exact-class crops for controlled replay. Test data are prohibited.
"""
from __future__ import annotations
import argparse,csv,json,math,random,shutil,sys
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from phase4_common import *
from evaluate_TCoreOther_comprehensive import box_iou,intersection_area

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--detector',type=Path,required=True); p.add_argument('--exact-initializer',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--det-conf-min',type=float,default=0.05); p.add_argument('--nms-iou',type=float,default=0.70); p.add_argument('--imgsz',type=int,default=1536); p.add_argument('--batch',type=int,default=4); p.add_argument('--device',default='0'); p.add_argument('--crop-margin',type=float,default=0.15); p.add_argument('--max-gt-iou',type=float,default=0.05); p.add_argument('--max-iop',type=float,default=0.10); p.add_argument('--max-background-per-image',type=int,default=6); p.add_argument('--low-exact-conf',type=float,default=0.55); p.add_argument('--low-exact-margin',type=float,default=0.15); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def norm(v): return ''.join(ch for ch in str(v).casefold() if ch.isalnum())

def main():
    a=parse_args(); out=a.output.expanduser().resolve()
    if out.exists():
        if not a.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    root,_,names=read_dataset(a.data)
    from ultralytics import YOLO
    detector=YOLO(str(a.detector.expanduser().resolve()),task='detect'); exact=YOLO(str(a.exact_initializer.expanduser().resolve()),task='classify')
    exact_names={int(k):str(v) for k,v in exact.names.items()}; by_name={norm(v):k for k,v in names.items()}; map_exact={k:by_name.get(norm(v)) for k,v in exact_names.items()}
    if any(v is None for v in map_exact.values()) or set(map_exact.values())!=set(names): raise RuntimeError(f'Exact classifier names mismatch: {exact_names} vs {names}')
    manifests=[]; summary={}
    for split in ('train','val'):
        images_root=split_dir(root,'images',split); labels_root=split_dir(root,'labels',split); images=list_images(images_root); split_rows=[]
        for start in range(0,len(images),a.batch):
            paths=images[start:start+a.batch]; det_results=detector.predict(source=[str(p) for p in paths],imgsz=a.imgsz,batch=min(a.batch,len(paths)),device=a.device,conf=a.det_conf_min,iou=a.nms_iou,max_det=300,verbose=False,save=False)
            for image,det_result in zip(paths,det_results):
                gt,size,_=image_gt(image,images_root,labels_root,names); relative=image.relative_to(images_root)
                with Image.open(image) as source: rgb=source.convert('RGB')
                gt_crops=[]; gt_meta=[]
                for index,item in enumerate(gt):
                    left,top,right,bottom=crop_box(item['bbox'],size,a.crop_margin)
                    if right<=left or bottom<=top: continue
                    crop=rgb.crop((left,top,right,bottom)); class_id=int(item['class_id']); class_name=names[class_id]
                    base=f"{'__'.join(relative.with_suffix('').parts)}__g{index:03d}.jpg"
                    binary_path=out/'pool'/'binary'/split/'ForeignMaterial'/base; exact_path=out/'pool'/'exact'/split/class_name/base
                    binary_path.parent.mkdir(parents=True,exist_ok=True); exact_path.parent.mkdir(parents=True,exist_ok=True); crop.save(binary_path,quality=94); crop.save(exact_path,quality=94)
                    gt_crops.append(crop); gt_meta.append((class_id,class_name,exact_path,binary_path,index))
                exact_results=exact.predict(source=gt_crops,imgsz=320,batch=min(32,len(gt_crops)),device=a.device,verbose=False) if gt_crops else []
                for meta,result in zip(gt_meta,exact_results):
                    class_id,class_name,exact_path,binary_path,index=meta; probs=result.probs.data.detach().float().cpu().numpy(); order=probs.argsort()[::-1]; top1=int(order[0]); top2=int(order[1]) if len(order)>1 else top1; pred=int(map_exact[top1]); conf=float(probs[top1]); margin=conf-float(probs[top2]); hard=(pred!=class_id or conf<a.low_exact_conf or margin<a.low_exact_margin)
                    split_rows.append({'split':split,'kind':'foreign_material','source_group':'ground_truth','source_image':str(image),'relative_image':relative.as_posix(),'crop_path':str(binary_path),'exact_crop_path':str(exact_path),'class_id':class_id,'class_name':class_name,'detector_confidence':'','exact_predicted_class':pred,'exact_confidence':conf,'exact_margin':margin,'exact_hard':hard})
                proposals=[]
                if det_result.boxes is not None and len(det_result.boxes):
                    boxes=det_result.boxes.xyxy.detach().float().cpu().numpy(); confs=det_result.boxes.conf.detach().float().cpu().numpy()
                    for pi,(box,conf) in enumerate(zip(boxes,confs)):
                        b=[float(v) for v in box]; area=max(0,b[2]-b[0])*max(0,b[3]-b[1]); max_iou=max((box_iou(b,x['bbox']) for x in gt),default=0.0); max_iop=max((intersection_area(b,x['bbox'])/area for x in gt),default=0.0) if area>0 else 1.0
                        if max_iou<=a.max_gt_iou and max_iop<=a.max_iop: proposals.append((float(conf),pi,b,max_iou,max_iop))
                proposals.sort(reverse=True); proposals=proposals[:a.max_background_per_image]
                group='negative_image' if not gt else 'positive_image_background'
                for conf,pi,b,max_iou,max_iop in proposals:
                    left,top,right,bottom=crop_box(b,size,a.crop_margin)
                    if right-left<16 or bottom-top<16: continue
                    base=f"{'__'.join(relative.with_suffix('').parts)}__p{pi:03d}.jpg"; path=out/'pool'/'binary'/split/'Background_NotFO'/group/base; path.parent.mkdir(parents=True,exist_ok=True); rgb.crop((left,top,right,bottom)).save(path,quality=94)
                    split_rows.append({'split':split,'kind':'background','source_group':group,'source_image':str(image),'relative_image':relative.as_posix(),'crop_path':str(path),'exact_crop_path':'','class_id':'','class_name':'Background_NotFO','detector_confidence':conf,'exact_predicted_class':'','exact_confidence':'','exact_margin':'','exact_hard':''})
            print(f'{split}: processed {min(start+a.batch,len(images))}/{len(images)}')
        manifests.extend(split_rows); summary[split]={"foreign_material_crops":sum(r['kind']=='foreign_material' for r in split_rows),"hard_exact_crops":sum(r['kind']=='foreign_material' and r['exact_hard'] for r in split_rows),"background_crops":sum(r['kind']=='background' for r in split_rows),"background_from_negative_images":sum(r['source_group']=='negative_image' for r in split_rows)}
    with (out/'crop_pool_manifest.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifests[0])); writer.writeheader(); writer.writerows(manifests)
    (out/'crop_pool_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
