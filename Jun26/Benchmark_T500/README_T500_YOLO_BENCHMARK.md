# T500 Final Benchmark — YOLOv8m vs YOLO11m

This package performs the complete model benchmark for the finalized T500 dataset.

## Frozen benchmark dataset

```text
C:\Hadi\Foreign_Material\Data\Ours\Jun-26_grouped_T500_split_80_10_10_v2
```

YAML:

```text
C:\Hadi\Foreign_Material\Data\Ours\Jun-26_grouped_T500_split_80_10_10_v2\data.yaml
```

Final classes:

| ID | Class |
|---:|---|
| 0 | BAW-Sack |
| 1 | Plastic |
| 2 | Paper_Cardboard |
| 3 | Wagon |
| 4 | Container |
| 5 | Other |

## Copy package to

```text
C:\Hadi\Foreign_Material\Code\Benchmark_T500
```

## Activate environment

```powershell
C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1
yolo checks
```

You should see CUDA and your RTX GPU.

## Step-by-step execution

### Step 1 — Smoke test

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\01_smoke_test.ps1
```

### Step 2 — Train YOLOv8m

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\02_train_yolov8m.ps1
```

### Step 3 — Train YOLO11m

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\03_train_yolo11m.ps1
```

### Step 4 — Evaluate both models

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\04_eval_compare.ps1
```

### Step 5 — Generate Report 2 draft

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\05_generate_report2.ps1
```

## Full runner

After smoke test succeeds, you may run:

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\run_full_benchmark.ps1
```

## Output root

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark
```

## Important outputs

Training:

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\yolov8m_T500_960
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\yolo11m_T500_960
```

Evaluation:

```text
C:\Hadi\Foreign_Material\Runs\T500_yolo_benchmark\comparison_eval
```

Expected evaluation files:

```text
comparison_overall.csv
comparison_per_class_map50_95.csv
comparison_summary.md
report2_T500_yolo_benchmark.md
```

Expected folders:

```text
yolov8m_T500_test
yolo11m_T500_test
yolov8m_T500_test_pred
yolo11m_T500_test_pred
```

Inside each validation folder, check:

```text
confusion_matrix.png
confusion_matrix_normalized.png
PR_curve.png
P_curve.png
R_curve.png
F1_curve.png
```

## Resume training if it crashes

YOLOv8m:

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\resume_yolov8m.ps1
```

YOLO11m:

```powershell
C:\Hadi\Foreign_Material\Code\Benchmark_T500\resume_yolo11m.ps1
```

## Scientific rule

Do not change:
- dataset split
- image size
- seed
- epochs
- confidence threshold
- IoU threshold

between YOLOv8m and YOLO11m.
