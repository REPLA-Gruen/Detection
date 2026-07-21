# How to Run — Final T500 YOLO Benchmark

## Step 0 — Copy package

Copy all files to:

```text
C:\Hadi\Foreign_Material\Code\Benchmark_T500
```

## Step 1 — Activate venv

```powershell
C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1
```

Check GPU:

```powershell
yolo checks
```

## Step 2 — Smoke test

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\01_smoke_test.ps1
```

Check:

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\smoke_yolov8m_T500
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\smoke_yolo11m_T500
```

## Step 3 — Train YOLOv8m

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\02_train_yolov8m.ps1
```

Expected output:

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\yolov8m_T500_960
```

## Step 4 — Train YOLO11m

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\03_train_yolo11m.ps1
```

Expected output:

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\yolo11m_T500_960
```

## Step 5 — Evaluate both

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\04_eval_compare.ps1
```

Expected output:

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval
```

## Step 6 — Generate Report 2 draft

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\05_generate_report2.ps1
```

Expected report:

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval\report2_T500_yolo_benchmark.md
```

## Resume if crashed

YOLOv8m:

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\resume_yolov8m.ps1
```

YOLO11m:

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\resume_yolo11m.ps1
```

## Files to send after benchmark

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval\comparison_overall.csv
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval\comparison_per_class_map50_95.csv
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval\comparison_summary.md
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval\report2_T500_yolo_benchmark.md
```

Also send:
- confusion matrices from both model folders
- PR curves
- several rendered prediction examples
