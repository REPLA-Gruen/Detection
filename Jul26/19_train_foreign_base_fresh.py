#!/usr/bin/env python3
"""Train A/B/C from the target dataset using generic Ultralytics pretraining only.

This is the clean rebuild route. It deliberately disables discovery of prior
project Phase-4/4B checkpoints so 'all-from-data' does not silently reuse old
project weights.
"""
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--overwrite',action='store_true');a=p.parse_args();c=load(a.config);c['pretrained']['stage_a_checkpoint']='';c['pretrained']['stage_b_checkpoint']='';c['pretrained']['stage_c_checkpoint']='';c['pretrained']['search_root']=str(ROOT/'__NO_PRIOR_PROJECT_WEIGHTS__');c['pretrained']['final_models_root']=str(ROOT/'__NO_PRIOR_PROJECT_WEIGHTS__');c['pretrained']['force_scratch']=False;c['pretrained']['use_generic_pretrained_when_selected_missing']=True
    tmp=ROOT/'_fresh_base_config.json';tmp.write_text(json.dumps(c,indent=2),encoding='utf-8')
    try:
        cmd=[sys.executable,str(ROOT/'03_train_base_models.py'),'--config',str(tmp)]
        if a.overwrite:cmd.append('--overwrite')
        return subprocess.run(cmd).returncode
    finally:
        try:tmp.unlink()
        except OSError:pass
if __name__=='__main__':raise SystemExit(main())
