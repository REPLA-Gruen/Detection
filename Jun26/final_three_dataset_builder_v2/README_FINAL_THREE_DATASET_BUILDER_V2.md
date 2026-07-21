# Final Three-Dataset Builder v2 — Balanced Task-Specific Splits

This is the final dataset-building package after restarting from the corrected Jun-26 raw analysis.

## Confirmed corrected raw dataset

From `01_analyze_yolo_dataset.py`:

```text
Images: 3359
Objects: 7403
Classes: 23
Missing label files: 0
Quality issues: 0
Raw imbalance ratio: 1859:1
```

## Final strategy

All datasets use the same clean benchmark universe:

```text
3334 images
```

The 25 images without a unique Wagon/Container context are excluded from all datasets.

However, each dataset receives its own optimized 80/10/10 split to maximize balance for its own task:

1. Dataset A — Detection-6
2. Dataset B1 — Wagon/Container Classification
3. Dataset B2 — Detection-4

This gives:
- same total image universe,
- task-specific balanced splits,
- clean comparison of strategies.

## Input

```text
C:\Hadi\Foreign_Material\Data\Ours\Jun-26
```

Expected:

```text
images
labels
data.yaml
```

## Output datasets

### Dataset A — Detection-6

```text
C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Final_A_Detection6_v2
```

Classes:

```text
0 BAW-Sack
1 Plastic
2 Paper_Cardboard
3 Wagon
4 Container
5 Other
```

### Dataset B1 — Classification WC

```text
C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Final_B1_Classification_WC_v2
```

Classes:

```text
Wagon
Container
```

### Dataset B2 — Detection-4

```text
C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Final_B2_Detection4_v2
```

Classes:

```text
0 BAW-Sack
1 Plastic
2 Paper_Cardboard
3 Other
```

Wagon and Container annotations are removed.

## Copy package to

```text
C:\Hadi\Foreign_Material\Code\Python\Jun26\final_three_dataset_builder_v2
```

## Run

```powershell
C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1
cd C:\Hadi\Foreign_Material\Code\Python\Jun26\final_three_dataset_builder_v2
.\run_final_three_dataset_builder_v2.ps1
```

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_final_three_dataset_builder_v2.ps1
```

## Report folder

```text
C:\Hadi\Foreign_Material\Runs\dataset_analysis\Jun-26_final_three_dataset_builder_v2
```

## Send these files back

```text
report_final_three_dataset_builder_v2.md
final_summary.csv
raw_corrected_summary.csv
benchmark_universe.csv
excluded_images.csv
dataset_A_detection6_distribution.csv
dataset_A_detection6_split_summary.csv
dataset_B1_classification_distribution.csv
dataset_B1_classification_split_summary.csv
dataset_B2_detection4_distribution.csv
dataset_B2_detection4_split_summary.csv
split_quality_scores.csv
quality_checks.csv
```
