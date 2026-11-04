#!/usr/bin/env python3
"""Compare BASE_FINAL_MODELS versus hybrid FINAL_MODELS on target validation."""
from __future__ import annotations
import argparse,json,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def fmt(v,T):return Path(str(v).replace('{T}',str(T))).expanduser().resolve()
def run_eval(config,model_root,data,output,split,overwrite):
    cmd=[sys.executable,str(ROOT/'10_test_evaluate_render.py'),'--config',str(config),'--model-root',str(model_root),'--data',str(data),'--split',split,'--output',str(output),'--render','none']
    if overwrite:cmd.append('--overwrite')
    subprocess.run(cmd,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--split',default='val');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load(a.config);T=int(c['threshold']);runs=fmt(c['runs_root'],T);data=fmt(c['dataset_output'],T)/'detection';root=runs/'PIPELINE_COMPARISON';base_out=root/f'base_{a.split}';hybrid_out=root/f'hybrid_{a.split}'
    run_eval(a.config,runs/'BASE_FINAL_MODELS',data,base_out,a.split,a.overwrite);run_eval(a.config,runs/'FINAL_MODELS',data,hybrid_out,a.split,a.overwrite)
    b=load(base_out/'reports'/'metrics.json');h=load(hybrid_out/'reports'/'metrics.json')
    def pick(m): return {'any_fo_recall':m['any_foreign_object_alarm']['recall'],'specificity':m['any_foreign_object_alarm']['specificity'],'balanced_accuracy':m['any_foreign_object_alarm']['balanced_accuracy'],'micro_f1':m['same_class_operational']['micro_f1'],'macro_f1':m['same_class_operational']['macro_f1'],'class_aware_f1':m['general_object_confusion']['class_aware_f1'],'geometric_localization_recall':m['general_object_confusion']['geometric_localization_recall']}
    pb,ph=pick(b),pick(h);delta={k:ph[k]-pb[k] for k in pb};result={'evaluation_domain':f'TARGET {a.split}','base':pb,'hybrid':ph,'delta':delta,'base_models':str(runs/'BASE_FINAL_MODELS'),'hybrid_models':str(runs/'FINAL_MODELS')};root.mkdir(parents=True,exist_ok=True);(root/f'comparison_{a.split}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
