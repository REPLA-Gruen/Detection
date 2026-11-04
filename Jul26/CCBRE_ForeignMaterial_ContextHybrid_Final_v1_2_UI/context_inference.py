#!/usr/bin/env python3
"""Runtime D1 Stage-0 context classification."""
from __future__ import annotations
from pathlib import Path
from typing import Any

def names_to_dict(names):
    if isinstance(names,dict):return {int(k):str(v) for k,v in names.items()}
    return {i:str(v) for i,v in enumerate(names)}

class ContextClassifier:
    def __init__(self,checkpoint,device='0',imgsz=640):
        from ultralytics import YOLO
        self.checkpoint=Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():raise FileNotFoundError(self.checkpoint)
        self.device=device;self.imgsz=int(imgsz);self.model=YOLO(str(self.checkpoint),task='classify');self.names=names_to_dict(self.model.names)
    def analyze(self,image)->dict[str,Any]:
        result=self.model.predict(source=str(image),imgsz=self.imgsz,device=self.device,verbose=False,save=False)[0]
        if result.probs is None:raise RuntimeError('Context classifier returned no probabilities')
        data=result.probs.data
        if hasattr(data,'detach'):values=data.detach().float().cpu().numpy().reshape(-1)
        else:
            import numpy as np;values=np.asarray(data,dtype=float).reshape(-1)
        top=int(values.argmax());ordered=values.argsort()[::-1]
        return {'class_id':top,'class_name':self.names[top],'confidence':float(values[top]),'probabilities':{self.names[i]:float(values[i]) for i in range(len(values))},'top2':[{'class_name':self.names[int(i)],'probability':float(values[int(i)])} for i in ordered[:min(2,len(ordered))]],'checkpoint':str(self.checkpoint)}
