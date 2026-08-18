#!/usr/bin/env python3
from __future__ import annotations
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    checks={}
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    checks['all_python_syntax']=True
    required=['19_train_foreign_base_fresh.py','20_create_D1_aligned.py','22_train_D1_context.py','24_evaluate_D1_context.py','25_import_existing_hybrid.py','26_consolidate_unified_models.py','27_verify_unified_product.py','28_test_complete_system.py','29_batch_analyze_folder.py','context_inference.py','combined_rendering.py','STANDALONE_APP.py','RUN_FINAL_PRODUCT.ps1','START_APP.ps1','README.txt']
    for name in required:checks[f'exists:{name}']=(ROOT/name).is_file()
    app=(ROOT/'STANDALONE_APP.py').read_text(encoding='utf-8');checks['app_has_no_permanent_model_selection_frame']='Model selection (optional)' not in app;checks['app_has_menu_models']="label='Models'" in app;checks['app_has_view_toggle']='F9' in app;checks['app_has_fullscreen']='F11' in app;checks['app_has_no_permanent_open_analyze_buttons']="ttk.Button(h,text='Open image'" not in app and "ttk.Button(h,text='Analyze'" not in app
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'));checks['context_config_present']='context' in cfg;checks['stageA_uavvaste_only_present']='stage_a_uavvaste_finetune' in cfg
    result={'passed':all(checks.values()),'checks':checks};print(json.dumps(result,indent=2));return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
