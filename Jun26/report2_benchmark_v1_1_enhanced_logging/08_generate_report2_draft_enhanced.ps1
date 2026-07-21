& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"
. "$PSScriptRoot\config_report2.ps1"
python "$ENV:FM_CODE\tools\generate_report2_draft_enhanced.py" --run_root "$ENV:RUN_ROOT"
