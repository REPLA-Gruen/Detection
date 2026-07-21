& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"

Set-Location C:\Hadi\Foreign_Material\Code\Benchmark_T500

Write-Host "================================================="
Write-Host "Evaluating YOLOv8m and YOLO11m on final T500 test split"
Write-Host "================================================="

python .\eval_compare_T500.py

Write-Host "Evaluation finished."
Write-Host "Check: C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval"
