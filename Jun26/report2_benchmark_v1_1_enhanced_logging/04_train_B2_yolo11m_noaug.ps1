& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"
. "$PSScriptRoot\config_report2.ps1"

python "$ENV:FM_CODE\tools\train_with_logging.py" `
  --run_root "$ENV:RUN_ROOT" `
  --exp_id "B2_yolo11m" `
  -- yolo detect train `
  model=yolo11m.pt `
  data="$ENV:DATA_B2_YAML" `
  epochs=$ENV:EPOCHS `
  imgsz=$ENV:IMG_SIZE `
  device=$ENV:DEVICE `
  batch=$ENV:BATCH `
  workers=$ENV:WORKERS `
  seed=$ENV:SEED `
  deterministic=True `
  patience=$ENV:PATIENCE `
  project="$ENV:RUN_ROOT\train" `
  name="B2_Detection4_yolo11m_noaug" `
  exist_ok=True `
  plots=True `
  save=True `
  val=True `
  rect=False `
  mosaic=0 mixup=0 copy_paste=0 `
  hsv_h=0 hsv_s=0 hsv_v=0 `
  degrees=0 translate=0 scale=0 shear=0 perspective=0 `
  flipud=0 fliplr=0 close_mosaic=0
