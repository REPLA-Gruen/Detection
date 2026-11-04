#!/usr/bin/env python3
"""Single orchestrator for the final Stage-0 + A/B/C pre-Phase-5 product."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def run(cmd,title):
    cmd=[str(x) for x in cmd];print('\n'+'='*92+'\n'+title+'\n'+'='*92+'\n'+' '.join(cmd));subprocess.run(cmd,check=True,env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')})
def target_create(c,config,overwrite):
    T=int(c['threshold']);d=c['dataset_creation'];cmd=[sys.executable,ROOT/'01_create_dataset.py','--yaml',*c['source_yamls'],'--output',fmt(c['dataset_output'],T),'--threshold',T,'--threshold-rule',d['threshold_rule'],'--dataset-policy',d['dataset_policy'],'--train-ratio',d['train_ratio'],'--val-ratio',d['val_ratio'],'--seed',c['seed'],'--split-trials',d['split_trials'],'--copy-mode',c['copy_mode']];cmd.append('--keep-exact-duplicates' if d['keep_exact_duplicates'] else '--no-keep-exact-duplicates');cmd.append('--write-masks' if d['write_masks'] else '--no-write-masks');
    if d.get('require_core_classes'):cmd+=['--require-core-class',*d['require_core_classes']]
    if overwrite:cmd.append('--overwrite')
    run(cmd,'CREATE TARGET T-CORE+OTHER')
def target_verify(c,config,overwrite):
    T=int(c['threshold']);root=fmt(c['dataset_output'],T)/'detection';cmd=[sys.executable,ROOT/'02_verify_dataset.py','--dataset-root',root,'--output',fmt(c['dataset_output'],T)/'verification','--mode','detection','--splits','train','val','test','--max-images-per-split',c['verification']['max_images_per_split'],'--seed',c['seed']];
    if overwrite:cmd.append('--overwrite')
    if c['verification'].get('fail_on_error',True):cmd.append('--fail-on-error')
    run(cmd,'VERIFY TARGET DATASET')
def stage(name,c,config,overwrite,download):
    if name=='check':run([sys.executable,ROOT/'00_check_final_inputs.py','--config',config],'CHECK INPUTS')
    elif name=='create-target':target_create(c,config,overwrite)
    elif name=='verify-target':target_verify(c,config,overwrite)
    elif name=='create-context':
        cmd=[sys.executable,ROOT/'20_create_D1_aligned.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'CREATE D1 CONTEXT DATASET ALIGNED TO T-CORE SPLITS')
    elif name=='verify-context':run([sys.executable,ROOT/'21_verify_D1_aligned.py','--config',config],'VERIFY D1 CONTEXT DATASET')
    elif name=='train-context':
        cmd=[sys.executable,ROOT/'22_train_D1_context.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'TRAIN D1 CONTEXT MODELS')
    elif name=='import-context':
        cmd=[sys.executable,ROOT/'23_import_existing_D1.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'IMPORT EXISTING D1 MODELS')
    elif name=='evaluate-context':
        cmd=[sys.executable,ROOT/'24_evaluate_D1_context.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'EVALUATE/SELECT D1 CONTEXT MODEL FROM VALIDATION')
    elif name=='train-foreign-base':
        cmd=[sys.executable,ROOT/'19_train_foreign_base_fresh.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'TRAIN TARGET-ONLY A/B/C BASE MODELS — GENERIC PRETRAINING, NO PRIOR PROJECT WEIGHTS')
    elif name=='import-phase4b-base':
        cmd=[sys.executable,ROOT/'04_import_existing_phase4b.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'IMPORT TARGET-ONLY PHASE-4B BASE A/B/C')
    elif name=='import-foreign':
        cmd=[sys.executable,ROOT/'25_import_existing_hybrid.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'IMPORT CURRENT FINAL HYBRID A/B/C')
    elif name=='prepare-uavvaste':
        cmd=[sys.executable,ROOT/'05_prepare_uavvaste.py','--config',config];cmd+=['--download'] if download else [];cmd+=['--overwrite'] if overwrite else [];run(cmd,'PREPARE UAVVASTE')
    elif name=='verify-uavvaste':
        cmd=[sys.executable,ROOT/'06_verify_uavvaste.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'VERIFY UAVVASTE')
    elif name=='finetune-stagea':
        cmd=[sys.executable,ROOT/'07_finetune_stageA_uavvaste.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'FINE-TUNE STAGE A ONLY WITH TARGET + UAVVASTE')
    elif name=='compare-stagea':run([sys.executable,ROOT/'08_compare_stageA.py','--config',config],'COMPARE BASE VS UAVVASTE STAGE A')
    elif name=='consolidate':
        cmd=[sys.executable,ROOT/'26_consolidate_unified_models.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'CONSOLIDATE STAGE 0 + A/B/C INTO ONE FINAL_MODELS')
    elif name=='verify-final':run([sys.executable,ROOT/'27_verify_unified_product.py','--config',config],'VERIFY UNIFIED FINAL PRODUCT')
    elif name=='compare-foreign':
        cmd=[sys.executable,ROOT/'11_compare_base_vs_hybrid.py','--config',config,'--split','val'];cmd+=['--overwrite'] if overwrite else [];run(cmd,'COMPARE FOREIGN BASE VS HYBRID')
    elif name=='test-complete':
        cmd=[sys.executable,ROOT/'28_test_complete_system.py','--config',config];cmd+=['--overwrite'] if overwrite else [];run(cmd,'TEST COMPLETE STAGE-0 + A/B/C PRODUCT')
    elif name=='app':run([sys.executable,ROOT/'STANDALONE_APP.py','--config',config],'START UNIFIED STANDALONE APP')
def main():
    choices=('check','create-target','verify-target','create-context','verify-context','train-context','import-context','evaluate-context','train-foreign-base','import-phase4b-base','import-foreign','prepare-uavvaste','verify-uavvaste','finetune-stagea','compare-stagea','consolidate','verify-final','compare-foreign','test-complete','app','all-existing','all-from-data')
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--stage',choices=choices,required=True);p.add_argument('--overwrite',action='store_true');p.add_argument('--download',action='store_true');a=p.parse_args();c=load(a.config)
    if a.stage=='all-existing':order=['check','create-context','verify-context','import-context','evaluate-context','import-foreign','consolidate','verify-final','test-complete']
    elif a.stage=='all-from-data':order=['create-target','verify-target','create-context','verify-context','train-context','evaluate-context','train-foreign-base','prepare-uavvaste','verify-uavvaste','finetune-stagea','compare-stagea','consolidate','verify-final','compare-foreign','test-complete']
    else:order=[a.stage]
    for s in order:stage(s,c,a.config,a.overwrite,a.download)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except subprocess.CalledProcessError as exc:print(f'ERROR: stage failed with exit code {exc.returncode}',file=sys.stderr);raise SystemExit(exc.returncode)
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
