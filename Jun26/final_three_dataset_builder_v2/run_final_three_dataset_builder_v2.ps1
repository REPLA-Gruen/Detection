& "C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1"

$CODE = "C:\Hadi\Foreign_Material\Code\Python\Jun26\final_three_dataset_builder_v2"
$RAW = "C:\Hadi\Foreign_Material\Data\Ours\Jun-26"
$YAML = "C:\Hadi\Foreign_Material\Data\Ours\Jun-26\data.yaml"

$OUT_A = "C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Final_A_Detection6_v2"
$OUT_B1 = "C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Final_B1_Classification_WC_v2"
$OUT_B2 = "C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Final_B2_Detection4_v2"

$REPORT = "C:\Hadi\Foreign_Material\Runs\dataset_analysis\Jun-26_final_three_dataset_builder_v2"

python "$CODE\final_three_dataset_builder_v2.py" `
  --dataset_root "$RAW" `
  --yaml "$YAML" `
  --out_a "$OUT_A" `
  --out_b1 "$OUT_B1" `
  --out_b2 "$OUT_B2" `
  --report_dir "$REPORT" `
  --train 0.80 `
  --val 0.10 `
  --test 0.10 `
  --seed 42 `
  --n_trials 5000 `
  --clear_outputs

Write-Host "DONE."
Write-Host "Datasets:"
Write-Host "$OUT_A"
Write-Host "$OUT_B1"
Write-Host "$OUT_B2"
Write-Host "Reports:"
Write-Host "$REPORT"
