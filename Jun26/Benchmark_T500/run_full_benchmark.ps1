& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"

$CODE = "C:\Hadi\Foreign_Material\Code\Benchmark_T500"

Write-Host "STEP 1: Train YOLOv8m"
& "$CODE\02_train_yolov8m.ps1"

Write-Host "STEP 2: Train YOLO11m"
& "$CODE\03_train_yolo11m.ps1"

Write-Host "STEP 3: Evaluate both models"
& "$CODE\04_eval_compare.ps1"

Write-Host "STEP 4: Generate Report 2"
& "$CODE\05_generate_report2.ps1"

Write-Host "FULL BENCHMARK COMPLETE."
