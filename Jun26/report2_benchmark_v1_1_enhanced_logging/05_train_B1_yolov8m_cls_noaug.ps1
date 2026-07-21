& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"
. "$PSScriptRoot\config_report2.ps1"

python "$ENV:FM_CODE\tools\train_with_logging.py" `
  --run_root "$ENV:RUN_ROOT" `
  --exp_id "B1_yolov8m_cls" `
  -- yolo classify train `
  model=yolov8m-cls.pt `
  data="$ENV:DATA_B1" `
  epochs=$ENV:EPOCHS `
  imgsz=$ENV:IMG_SIZE `
  device=$ENV:DEVICE `
  batch=$ENV:BATCH `
  workers=$ENV:WORKERS `
  seed=$ENV:SEED `
  deterministic=True `
  patience=$ENV:PATIENCE `
  project="$ENV:RUN_ROOT\train" `
  name="B1_Classification_yolov8m_cls_noaug" `
  exist_ok=True `
  plots=True `
  save=True `
  val=True `
  rect=False `
  erasing=0 `
  flipud=0 fliplr=0 `
  hsv_h=0 hsv_s=0 hsv_v=0 `
  degrees=0 translate=0 scale=0 shear=0 perspective=0
