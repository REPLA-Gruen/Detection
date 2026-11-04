#!/usr/bin/env python3
"""Train the unified D1 context models on the aligned context dataset."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def run(cmd,title):
    print('\n'+'='*88+'\n'+title+'\n'+'='*88+'\n'+' '.join(str(x) for x in cmd));subprocess.run([str(x) for x in cmd],check=True,env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')})
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);data=fmt(c['context']['dataset_output'],T);runs=fmt(c['runs_root'],T);project=runs/'CONTEXT_TRAINING_RUNS';s=c['context']['training']
    if not (data/'train'/'container').is_dir():raise FileNotFoundError(f'Create D1 first: {data}')
    candidates=[('yolo11m',ROOT/'train_D1_yolo11.py',s['yolo11_pretrained'],s['yolo11_run_name'])]
    if c['context'].get('train_reference_yolov8',True):candidates.append(('yolov8m',ROOT/'train_D1_yolov8.py',s['yolov8_pretrained'],s['yolov8_run_name']))
    manifest={'dataset':str(data),'project':str(project),'models':{}}
    for family,script,pretrained,name in candidates:
        run_dir=project/name;best=run_dir/'weights'/'best.pt'
        if best.is_file() and not a.overwrite:
            print(f'Reusing completed context model: {best}')
        else:
            if run_dir.exists() and a.overwrite:shutil.rmtree(run_dir)
            cmd=[sys.executable,script,'--data',data,'--model',pretrained,'--project',project,'--name',name,'--epochs',s['epochs'],'--patience',s['patience'],'--imgsz',s['imgsz'],'--batch',s['batch'],'--device',c['device'],'--workers',c['workers'],'--seed',c['seed'],'--optimizer',s['optimizer'],'--lr0',s['lr0'],'--lrf',s['lrf'],'--momentum',s['momentum'],'--weight-decay',s['weight_decay'],'--warmup-epochs',s['warmup_epochs'],'--dropout',s['dropout'],'--augmentation',s['augmentation']]
            if not s.get('amp',True):cmd.append('--no-amp')
            run(cmd,f'TRAIN D1 CONTEXT — {family}')
        if not best.is_file():raise FileNotFoundError(best)
        manifest['models'][family]=str(best.resolve())
    project.mkdir(parents=True,exist_ok=True);(project/'context_training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except subprocess.CalledProcessError as exc:print(f'ERROR: training command failed: {exc.returncode}',file=sys.stderr);raise SystemExit(exc.returncode)
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
