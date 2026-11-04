#!/usr/bin/env python3
"""Audit and rank hard positive training images for Any-FM detector replay."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from phase4_common import *
from evaluate_TCoreOther_comprehensive import box_iou,intersection_area

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--detector",type=Path,required=True); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",default="train"); p.add_argument("--det-conf-min",type=float,default=0.05); p.add_argument("--low-conf",type=float,default=0.25); p.add_argument("--poor-iou",type=float,default=0.20)
    p.add_argument("--imgsz",type=int,default=1536); p.add_argument("--batch",type=int,default=4); p.add_argument("--device",default="0"); p.add_argument("--nms-iou",type=float,default=0.70); return p.parse_args()

def main():
    a=parse_args(); out=a.output.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    root,_,names=read_dataset(a.data); images_root=split_dir(root,"images",a.split); labels_root=split_dir(root,"labels",a.split); images=list_images(images_root)
    from ultralytics import YOLO
    model=YOLO(str(a.detector.expanduser().resolve()),task='detect'); rows=[]; image_rows=[]; strata=Counter()
    for start in range(0,len(images),a.batch):
        paths=images[start:start+a.batch]; results=model.predict(source=[str(p) for p in paths],imgsz=a.imgsz,batch=min(a.batch,len(paths)),device=a.device,conf=a.det_conf_min,iou=a.nms_iou,max_det=300,verbose=False,save=False)
        for image,result in zip(paths,results):
            gt,size,label=image_gt(image,images_root,labels_root,names)
            if not gt: continue
            predictions=[]
            if result.boxes is not None and len(result.boxes):
                boxes=result.boxes.xyxy.detach().float().cpu().numpy(); confs=result.boxes.conf.detach().float().cpu().numpy()
                predictions=[{"bbox":[float(v) for v in b],"confidence":float(c)} for b,c in zip(boxes,confs)]
            hardness=0.0; hard_objects=0
            for gi,item in enumerate(gt):
                overlaps=[]
                for pred in predictions:
                    inter=intersection_area(item['bbox'],pred['bbox'])
                    if inter>0: overlaps.append((box_iou(item['bbox'],pred['bbox']),pred['confidence']))
                if overlaps:
                    best_iou,best_conf=max(overlaps,key=lambda x:(x[0],x[1])); missed=False
                else:
                    best_iou=best_conf=0.0; missed=True
                reasons=[]; score=0.0
                if missed: reasons.append('missed'); score+=3.0
                if best_iou<a.poor_iou: reasons.append('poor_iou'); score+=2.0
                if best_conf<a.low_conf: reasons.append('low_confidence'); score+=1.0
                sbin=size_bin(item['bbox'],size)
                if sbin=='small': reasons.append('small'); score+=0.5
                is_hard=bool(reasons); hardness+=score; hard_objects+=int(is_hard)
                class_id=int(item['class_id']); context=context_from_filename(image)
                rows.append({"image":str(image),"relative_image":image.relative_to(images_root).as_posix(),"gt_index":gi,"class_id":class_id,"class_name":names[class_id],"size_bin":sbin,"context":context,"best_iou":best_iou,"best_confidence":best_conf,"missed":missed,"hard":is_hard,"reasons":";".join(reasons),"hardness_score":score})
                if is_hard: strata[(names[class_id],sbin,context)]+=1
            image_rows.append({"image":str(image),"relative_image":image.relative_to(images_root).as_posix(),"instances":len(gt),"hard_objects":hard_objects,"image_hardness":hardness,"classes":";".join(sorted({names[int(x['class_id'])] for x in gt})),"context":context_from_filename(image)})
        print(f"processed {min(start+a.batch,len(images))}/{len(images)}")
    image_rows.sort(key=lambda x:(x['image_hardness'],x['hard_objects'],x['instances']),reverse=True)
    for filename,data in [('hard_positive_objects.csv',rows),('hard_positive_images.csv',image_rows)]:
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    summary={"positive_images":len(image_rows),"objects":len(rows),"hard_objects":sum(r['hard'] for r in rows),"images_with_hard_objects":sum(r['hard_objects']>0 for r in image_rows),"hard_strata":[{"class":k[0],"size":k[1],"context":k[2],"count":v} for k,v in strata.most_common()]}
    (out/'hard_positive_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f"ERROR: {exc}",file=sys.stderr); raise SystemExit(1)
