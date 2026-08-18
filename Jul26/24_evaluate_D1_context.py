#!/usr/bin/env python3
"""Evaluate D1 candidates on aligned val/test and select deployment context model from validation only."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def run(cmd,title):
    print('\n'+'='*88+'\n'+title+'\n'+'='*88+'\n'+' '.join(str(x) for x in cmd));subprocess.run([str(x) for x in cmd],check=True,env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')})
def candidates(c,runs):
    s=c['context']['training'];out={}
    trained=runs/'CONTEXT_TRAINING_RUNS'
    for family,name in [('yolo11m',s['yolo11_run_name']),('yolov8m',s['yolov8_run_name'])]:
        p=trained/name/'weights'/'best.pt'
        if p.is_file():out[family]={'path':p,'provenance':'trained_on_unified_aligned_D1'}
    imported=runs/'CONTEXT_IMPORTED_MODELS'
    for family,folder in [('yolo11m','YOLO11m'),('yolov8m','YOLOv8m')]:
        p=imported/folder/'best.pt'
        if family not in out and p.is_file():out[family]={'path':p,'provenance':'legacy_import'}
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);runs=fmt(c['runs_root'],T);data=fmt(c['context']['dataset_output'],T);cand=candidates(c,runs)
    if not cand:raise FileNotFoundError('No D1 models. Run train-context or import-context.')
    root=runs/'CONTEXT_EVALUATION';all_metrics={}
    for family,item in cand.items():
        all_metrics[family]={'checkpoint':str(item['path']),'provenance':item['provenance']}
        for split in ('val','test'):
            out=root/family/split
            if out.exists() and a.overwrite:shutil.rmtree(out)
            if not (out/'reports'/'metrics.json').is_file():
                cmd=[sys.executable,ROOT/'evaluate_D1_classifier.py','--model',item['path'],'--data',data,'--split',split,'--output',out,'--imgsz',c['context']['training']['imgsz'],'--batch',c['context']['evaluation']['batch'],'--device',c['device'],'--positive-class',c['context']['evaluation']['positive_class'],'--render',c['context']['evaluation']['render']]
                if a.overwrite:cmd.append('--overwrite')
                run(cmd,f'EVALUATE D1 {family} — {split}')
            all_metrics[family][split]=load(out/'reports'/'metrics.json')
    floor=float(c['context']['minimum_validation_accuracy']);priority=c['context']['deployment_priority'];selected=None
    checks={}
    for family in priority:
        if family not in all_metrics:continue
        acc=float(all_metrics[family]['val']['accuracy']);checks[family]={'validation_accuracy':acc,'passes_floor':acc>=floor}
        if selected is None and acc>=floor:selected=family
    if selected is None:
        # Do not choose based on test. Highest validation accuracy is recorded as engineering fallback only.
        selected=max(all_metrics,key=lambda f:float(all_metrics[f]['val']['accuracy']))
        accepted=False
    else:accepted=True
    decision={'selection_basis':'validation only; test metrics are confirmatory and are not used to switch the selected model','deployment_priority':priority,'minimum_validation_accuracy':floor,'checks':checks,'selected_family':selected,'selected_checkpoint':all_metrics[selected]['checkpoint'],'selected_provenance':all_metrics[selected]['provenance'],'accepted':accepted,'test_independence_note':'For models trained by this unified workflow, the aligned test split is untouched by D1 fitting/selection. For legacy imported models, test independence relative to their historical training split is not guaranteed.','metrics':all_metrics}
    sel=runs/'CONTEXT_SELECTION';sel.mkdir(parents=True,exist_ok=True);(sel/'selected_context_model.json').write_text(json.dumps(decision,indent=2),encoding='utf-8');print(json.dumps(decision,indent=2));return 0 if accepted else 2
if __name__=='__main__':
    try:raise SystemExit(main())
    except subprocess.CalledProcessError as exc:print(f'ERROR: evaluation failed: {exc.returncode}',file=sys.stderr);raise SystemExit(exc.returncode)
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
