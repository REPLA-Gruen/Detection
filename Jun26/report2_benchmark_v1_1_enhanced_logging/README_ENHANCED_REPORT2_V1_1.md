# Report 2 Benchmark Package v1.1 — Enhanced Logging and Evaluation

This version keeps the same no-augmentation baseline, but adds automatic collection of:

- total training time
- average epoch time
- best epoch
- final learning rate
- model size
- parameter count
- GFLOPs parsed from logs
- inference latency
- confusion matrix file paths
- per-class AP tables
- unified comparison tables

## What changed compared with v1

Training scripts now call:

```text
tools\train_with_logging.py
```

This wrapper runs the same Ultralytics command, streams the console output, saves the full log, measures wall-clock training time, and writes a timing JSON.

Evaluation now uses:

```text
tools\unified_eval_report2_enhanced.py
```

It evaluates all available trained models on the test split, saves metrics, extracts inference speed, writes per-class tables, and records confusion matrix image paths.

## Copy to

```text
C:\Hadi\Foreign_Material\Code\Python\Jun26\report2_benchmark_v1_1_enhanced_logging
```

## Run

```powershell
C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1
cd C:\Hadi\Foreign_Material\Code\Python\Jun26\report2_benchmark_v1_1_enhanced_logging

.\00_smoke_test.ps1
.\01_train_A_yolov8m_noaug.ps1
.\02_train_A_yolo11m_noaug.ps1
.\03_train_B2_yolov8m_noaug.ps1
.\04_train_B2_yolo11m_noaug.ps1
.\05_train_B1_yolov8m_cls_noaug.ps1
.\06_train_B1_yolo11m_cls_noaug.ps1
.\07_unified_eval_noaug_enhanced.ps1
.\08_generate_report2_draft_enhanced.ps1
```

## Main outputs

```text
C:\Hadi\Foreign_Material\Runs\Report2_Benchmark_v1_1_noaug_enhanced\logs
C:\Hadi\Foreign_Material\Runs\Report2_Benchmark_v1_1_noaug_enhanced\summary_tables\unified_metrics_enhanced.csv
C:\Hadi\Foreign_Material\Runs\Report2_Benchmark_v1_1_noaug_enhanced\summary_tables\training_efficiency_summary.csv
C:\Hadi\Foreign_Material\Runs\Report2_Benchmark_v1_1_noaug_enhanced\summary_tables\per_class_detection_metrics.csv
C:\Hadi\Foreign_Material\Runs\Report2_Benchmark_v1_1_noaug_enhanced\report2_draft_noaug_enhanced.md
```
