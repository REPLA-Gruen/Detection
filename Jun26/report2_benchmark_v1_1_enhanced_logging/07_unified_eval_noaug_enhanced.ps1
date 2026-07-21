& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"
. "$PSScriptRoot\config_report2.ps1"

python "$ENV:FM_CODE\tools\unified_eval_report2_enhanced.py" `
  --run_root "$ENV:RUN_ROOT" `
  --data_a_yaml "$ENV:DATA_A_YAML" `
  --data_b1 "$ENV:DATA_B1" `
  --data_b2_yaml "$ENV:DATA_B2_YAML" `
  --imgsz $ENV:IMG_SIZE `
  --device $ENV:DEVICE
