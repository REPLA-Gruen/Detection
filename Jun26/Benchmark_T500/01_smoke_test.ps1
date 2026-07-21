& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"

Set-Location C:\Hadi\Foreign_Material

$DATA = "C:\Hadi\Foreign_Material\Data\Ours\Jun-26_grouped_T500_split_80_10_10_v2\data.yaml"
$PROJECT = "C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark"

Write-Host "================================================="
Write-Host "Smoke test YOLOv8m on T500 benchmark"
Write-Host "================================================="

yolo detect train `
    model=yolov8m.pt `
    data=$DATA `
    imgsz=960 `
    epochs=1 `
    batch=4 `
    device=0 `
    workers=2 `
    seed=42 `
    pretrained=True `
    project=$PROJECT `
    name=smoke_yolov8m_T500 `
    exist_ok=True

Write-Host "================================================="
Write-Host "Smoke test YOLO11m on T500 benchmark"
Write-Host "================================================="

yolo detect train `
    model=yolo11m.pt `
    data=$DATA `
    imgsz=960 `
    epochs=1 `
    batch=4 `
    device=0 `
    workers=2 `
    seed=42 `
    pretrained=True `
    project=$PROJECT `
    name=smoke_yolo11m_T500 `
    exist_ok=True

Write-Host "Smoke tests finished."
