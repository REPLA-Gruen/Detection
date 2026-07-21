& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"

Set-Location C:\Hadi\Foreign_Material\Code\Benchmark_T500

python .\generate_report2_T500.py

Write-Host "Report 2 draft generated."
Write-Host "Check: C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval\report2_T500_yolo_benchmark.md"
