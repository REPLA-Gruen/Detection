#!/usr/bin/env python3
"""Build target+UAVVaste Stage-A train view and fine-tune ONLY Stage A.

The target validation split remains target-only. Stage B/C are never trained here.
"""
from __future__ import annotations
import argparse, csv, json, random, shutil, sys
from pathlib import Path
import yaml
from ultralytics_runtime import suppress_auxiliary_amp_model_check
ROOT=Path(__file__).resolve().parent
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T): return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def images(folder): return sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.casefold() in EXTS)
def transfer(src,dst,mode='hardlink'):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): dst.unlink()
    if mode=='hardlink':
        try: dst.hardlink_to(src); return
        except OSError: pass
    shutil.copy2(src,dst)
def oneclass(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True); lines=[]
    if src.is_file():
        for n,raw in enumerate(src.read_text(encoding='utf-8-sig').splitlines(),1):
            p=raw.strip().split();
            if not p: continue
            if len(p)!=5: raise ValueError(f'Expected bbox YOLO: {src}:{n}')
            lines.append('0 '+' '.join(p[1:]))
    dst.write_text(('\n'.join(lines)+'\n') if lines else '',encoding='utf-8')
def resolve_target_view(c,runs):
    local=runs/'TRAINING_ASSETS'/'StageA_Final_Replay_View'
    if (local/'dataset.yaml').is_file(): return local
    existing=Path(c['existing_phase4b']['stage_a_training_view']).expanduser().resolve()
    if (existing/'dataset.yaml').is_file(): return existing
    raise FileNotFoundError('No StageA_Final_Replay_View found. Run train-base or import existing 4B assets.')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'config.json'); p.add_argument('--overwrite',action='store_true'); a=p.parse_args(); c=load(a.config); T=int(c['threshold']); runs=fmt(c['runs_root'],T); base=runs/'BASE_FINAL_MODELS'/'StageA_AnyFM_Detector'/'best.pt'; uav=Path(c['uavvaste']['prepared_yolo']).expanduser().resolve(); target=resolve_target_view(c,runs); mixed=runs/'TRAINING_ASSETS'/'StageA_TargetPlusUAVVaste'; s=c['stage_a_uavvaste_finetune']
    if not base.is_file(): raise FileNotFoundError(f'Base Stage A missing: {base}')
    if not (uav/'dataset.yaml').is_file(): raise FileNotFoundError('Prepare UAVVaste first.')
    if mixed.exists():
        if a.overwrite: shutil.rmtree(mixed)
        else: print(f'Reusing mixed Stage-A view: {mixed}')
    if not mixed.exists():
        manifest=[]; target_train=0
        for split in ('train','val'):
            ir=target/'images'/split; lr=target/'labels'/split; ims=images(ir)
            for idx,img in enumerate(ims):
                rel=img.relative_to(ir); dst=mixed/'images'/split/'target'/rel; transfer(img,dst,c['copy_mode']); src_label=(lr/rel).with_suffix('.txt'); dst_label=(mixed/'labels'/split/'target'/rel).with_suffix('.txt'); oneclass(src_label,dst_label); manifest.append({'split':split,'domain':'target','source':str(img),'output':str(dst)})
            if split=='train': target_train=len(ims)
        ext=images(uav/'images'/'train'); rng=random.Random(c['seed']); rng.shuffle(ext); requested=round(target_train*float(s['external_train_ratio_to_target_images'])); n=min(len(ext),requested,int(s['maximum_external_train_images'])); selected=ext[:n]
        for idx,img in enumerate(selected):
            rel=Path(f'{idx:06d}__{img.name}'); dst=mixed/'images'/'train'/'uavvaste'/rel; transfer(img,dst,c['copy_mode']); oneclass(uav/'labels'/'train'/img.with_suffix('.txt').name,(mixed/'labels'/'train'/'uavvaste'/rel).with_suffix('.txt')); manifest.append({'split':'train','domain':'UAVVaste','source':str(img),'output':str(dst)})
        (mixed/'dataset.yaml').write_text(yaml.safe_dump({'path':str(mixed),'train':'images/train','val':'images/val','names':{0:'ForeignMaterial'}},sort_keys=False),encoding='utf-8')
        with (mixed/'mix_manifest.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            w=csv.DictWriter(stream,fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
        (mixed/'mix_summary.json').write_text(json.dumps({'target_train_images':target_train,'uavvaste_train_images_added':n,'external_ratio_actual':n/max(1,target_train),'validation_policy':'target-only validation'},indent=2),encoding='utf-8')
    from ultralytics import YOLO
    suppress_auxiliary_amp_model_check(bool(s.get('skip_ultralytics_amp_check',True)))
    project=runs/'UAVVASTE_FINETUNE'/'STAGEA'; name='StageA_4B_Finetuned_TargetPlusUAVVaste'; run=project/name
    if run.exists() and a.overwrite: shutil.rmtree(run)
    print('='*88); print('STAGE-A UAVVASTE FINE-TUNING'); print(f'Initializer: {base}'); print(f'Target view: {target}'); print(f'Mixed view: {mixed}'); print('Stage B/C: unchanged'); print('='*88)
    model=YOLO(str(base),task='detect')
    result=model.train(data=str(mixed/'dataset.yaml'),project=str(project),name=name,exist_ok=True,imgsz=int(s['imgsz']),epochs=int(s['epochs']),patience=int(s['patience']),batch=int(s['batch']),device=c['device'],workers=int(c['workers']),seed=int(c['seed']),deterministic=True,optimizer=s['optimizer'],lr0=float(s['lr0']),lrf=float(s['lrf']),momentum=float(s['momentum']),weight_decay=float(s['weight_decay']),warmup_epochs=float(s['warmup_epochs']),mosaic=float(s['mosaic']),close_mosaic=int(s['close_mosaic']),degrees=float(s['degrees']),translate=float(s['translate']),scale=float(s['scale']),hsv_h=float(s['hsv_h']),hsv_s=float(s['hsv_s']),hsv_v=float(s['hsv_v']),fliplr=float(s['fliplr']),flipud=0.0,mixup=0.0,copy_paste=0.0,amp=bool(s.get('amp',True)),plots=True,val=True,max_det=300)
    save=Path(getattr(result,'save_dir',run)); best=save/'weights'/'best.pt';
    if not best.is_file(): raise FileNotFoundError(best)
    (save/'uavvaste_finetune_manifest.json').write_text(json.dumps({'initializer':str(base),'target_view':str(target),'mixed_view':str(mixed),'candidate':str(best),'stage_b_unchanged':True,'stage_c_unchanged':True},indent=2),encoding='utf-8'); print(f'Candidate: {best}'); return 0
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc: print(f'ERROR: {exc}',file=sys.stderr); raise SystemExit(1)
