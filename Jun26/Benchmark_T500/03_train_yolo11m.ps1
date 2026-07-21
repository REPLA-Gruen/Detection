& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"

Set-Location C:\Hadi\Foreign_Material

$DATA = "C:\Hadi\Foreign_Material\Data\Ours\Jun-26_grouped_T500_split_80_10_10_v2\data.yaml"
$PROJECT = "C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark"
$NAME = "yolo11m_T500_960"

Write-Host "================================================="
Write-Host "Training YOLO11m on final T500 benchmark"
Write-Host "================================================="
Write-Host "Dataset: $DATA"
Write-Host "Output : $PROJECT\$NAME"

yolo detect train `
    model=yolo11m.pt `
    data=$DATA `
    imgsz=960 `
    epochs=100 `
    batch=16 `
    device=0 `
    workers=8 `
    seed=42 `
    patience=50 `
    pretrained=True `
    project=$PROJECT `
    name=$NAME `
    exist_ok=True

Write-Host "YOLO11m training finished."
