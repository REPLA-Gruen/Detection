#!/usr/bin/env python3
"""One command to test Stage 0 context plus the complete A/B/C foreign-material pipeline."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def run(cmd,title):
    print('\n'+'='*88+'\n'+title+'\n'+'='*88+'\n'+' '.join(str(x) for x in cmd));subprocess.run([str(x) for x in cmd],check=True,env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':os.environ.get('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')})
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);runs=fmt(c['runs_root'],T);final=runs/'FINAL_MODELS';manifest=load(final/'final_model_manifest.json');output=runs/'COMPLETE_SYSTEM_TEST'
    if output.exists() and a.overwrite:shutil.rmtree(output)
    output.mkdir(parents=True,exist_ok=True)
    context_out=output/'CONTEXT_D1_TEST';foreign_out=output/'FOREIGN_MATERIAL_TEST'
    cmd=[sys.executable,ROOT/'evaluate_D1_classifier.py','--model',manifest['models']['context'],'--data',fmt(c['context']['dataset_output'],T),'--split','test','--output',context_out,'--imgsz',c['context']['training']['imgsz'],'--batch',c['context']['evaluation']['batch'],'--device',c['device'],'--positive-class',c['context']['evaluation']['positive_class'],'--render','errors']
    if a.overwrite:cmd.append('--overwrite')
    run(cmd,'TEST STAGE 0 — CONTEXT CLASSIFICATION')
    cmd=[sys.executable,ROOT/'10_test_evaluate_render.py','--config',a.config,'--model-root',final,'--split','test','--output',foreign_out]
    if a.overwrite:cmd.append('--overwrite')
    run(cmd,'TEST STAGES A/B/C — FOREIGN MATERIAL')
    context_metrics=load(context_out/'reports'/'metrics.json');foreign_metrics=load(foreign_out/'reports'/'metrics.json')
    summary={'product':'CCBRE Context + Hybrid Foreign-Material Final Product','context_test':{'accuracy':context_metrics.get('accuracy'),'balanced_accuracy':context_metrics.get('balanced_accuracy'),'macro_f1':context_metrics.get('macro_f1'),'mcc':context_metrics.get('matthews_correlation_coefficient'),'images':context_metrics.get('images'),'provenance_note':'Independent only for a context model trained/selected on this aligned split design; legacy imported D1 models retain their historical evidence limitation.'},'foreign_material_test':{'images':foreign_metrics.get('counts',{}).get('images'),'any_fo':foreign_metrics.get('any_foreign_object_alarm'),'same_class_operational':foreign_metrics.get('same_class_operational'),'general_object_confusion':foreign_metrics.get('general_object_confusion')},'causal_note':'Stage 0 is recorded as acquisition-context metadata in this pre-Phase-5 product; it does not gate or suppress A/B/C inference. Therefore no causal context-routing gain is claimed.'}
    (output/'COMPLETE_SYSTEM_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except subprocess.CalledProcessError as exc:print(f'ERROR: test stage failed: {exc.returncode}',file=sys.stderr);raise SystemExit(exc.returncode)
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
