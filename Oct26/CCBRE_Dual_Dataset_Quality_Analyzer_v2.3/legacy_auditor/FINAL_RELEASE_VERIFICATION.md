# CCBRE Oct-26 Auditor — v1.7 validated release (source frozen)

**Release status:** The original v1.7 Python analyzer, its extension module, the three existing PowerShell launchers and five existing tests are **unchanged**. This ZIP adds only an independent result verifier (`verify_v17_results.py`, `RUN_VERIFY_RESULTS.ps1`), an optional explicit class mapping and this evidence record. It does not rewrite or alter the scientific method used in the October 30 full run.

## Evidence used for this check

- Uploaded full standalone output folder: `Analysis_v1.7`, generated October 30, 2026.
- Historical v1.6 output was used only to check whether the existing inventory and image fingerprints changed unexpectedly. It is **not** required for standalone execution.
- Five existing regression scripts passed in this environment.
- The added **standard-library, read-only verifier** passed **56/56 checks** on the uploaded full results using the `--expected-snapshot` option. It recomputes counts from CSV rows rather than trusting the JSON summary alone.

## Verified counts, meaning, and evidence limitations

| Measure | Full v1.7 result |
| --- | ---: |
| October image records | 9,628 |
| Unique byte-identical image content groups | 9,407 |
| Foreign material boxes / polygons | 14,087 / 14,087 |
| Collection / floorspread records | 7,463 / 2,165 |
| Positive / explicitly empty image records | 4,787 / 4,841 |
| Exact duplicate-image groups | 221 |
| Different box signatures in duplicate groups | 135 |
| July source images matched in October | 3,359 / 3,359 |
| October records with exact July image bytes | 3,579 (includes 220 duplicate versions) |
| October records without exact/pixel identity to July | 6,049 |
| Distinct *new byte-content groups* (not independent acquisitions) | 6,048 |
| Day-level split groups / largest group | 3,236 / 920 |
| July YOLO rows | 28,168 |
| July context rows | 22,528 |
| July foreign-material rows | 5,640 |
| Mapped / unresolved July foreign-material rows | 5,427 / 213 |
| October box counts on same July images (possible annotation versions) | 3,860–3,981 |
| July-mapped boxes matched by class-aware IoU >= 0.50 | 3,565–3,708 |
| July-reference label-diff issues | 0 |

*Important:* The `03_class_statistics.csv` file contains **both context-specific rows and an `all` total**. Do not sum all rows, which would double-count objects. The verifier checks the `all` total separately from the combined two contexts.

*Important:* Per-class minimum/maximum match columns (`45_reference_class_diff_counts.csv`) may not add to the **global** minimum/maximum in `39_reference_label_diff_summary.json`. Per-class extrema can belong to different versions of the same image. Use the per-image totals from file 40 for overall bounds.

*Important:* The 221 exact pairs are same-file copies. The 196 close dHash links (distance <= 6) are **perceptual candidates only**, not verified duplicates. Do not use the v1.6/v1.7 dHash-extended groups directly as train/test split IDs.

## Final code: standalone run sequence

From the extracted package in PowerShell:

```powershell
py -3 -m pip install -r requirements.txt
py -3 .\test_smoke.py
py -3 .\test_v14_features.py
py -3 .\test_v15_regression.py
py -3 .\test_v16_regression.py
py -3 .\test_v17_regression.py
```

Step 1. Generate July reference **with this same v1.7 package** from the authoritative July source photographs:

```powershell
.\RUN_JULY_FINGERPRINT.ps1 `
  -JulyImageFolders @('C:\ACTUAL\PATH\TO\JULY\images') `
  -Output 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7'
```

Confirm the manifest has **3,359 original photographs**. Do not include derived training copies.

Step 2. Run full October auditing and July label reconciliation (also from this same package):

```powershell
.\RUN_ANALYSIS.ps1 `
  -Dataset 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw' `
  -Output 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7' `
  -ReferenceManifest 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7\31_fingerprint_manifest.csv' `
  -ReferenceLabels 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\labels' `
  -ReferenceYaml 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\dataset.yaml'
```

Step 3. Verify all published figures **independently**:

```powershell
.\RUN_VERIFY_RESULTS.ps1 -ExpectedSnapshot
```

The verifier has no raw-image or previous-code dependency. Omit `-ExpectedSnapshot` if the upstream S3 export or local source population has genuinely changed since this snapshot.

### Optional class-map second pass

The v1.7 default intentionally leaves two July names unmapped: `Diverses` (200 boxes) and `Styropor/Dämmung` (13 boxes). The **exporter metadata explicitly maps `Styropor/Dämmung` to `Insulation`**, but the frozen v1.7 implementation requires a manual two-column CSV to admit it. A ready-to-review `reference_class_map_styropor_optional.csv` is included. Re-run the label diff in a *new output directory*:

```powershell
.\RUN_LABEL_DIFF_ONLY.ps1 `
  -AnalysisInput 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7' `
  -Output 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\LabelDiff_StyroporMapped' `
  -ReferenceManifest 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7\31_fingerprint_manifest.csv' `
  -ReferenceLabels 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\labels' `
  -ReferenceYaml 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\dataset.yaml' `
  -ReferenceClassMap '.\reference_class_map_styropor_optional.csv'
```

This will shift the **class-policy mapping totals** from 5,427 mapped + 213 unmapped to **5,440 mapped + 200 unmapped**. It will also recalculate matches, which **must be read from that new run** rather than inferred. `Diverses` remains excluded pending exporter-owner confirmation.

## Remaining scientific gates (outside the code)

1. The annotation owner must determine whether the substantially reduced October annotations on July photographs are a deliberate quality-control revision or accidental omission. This is the main barrier to a canonical November dataset and comparable model evaluation.
2. G3: Human reviewer must sign the EXIF-orientation review ledger (reviewer and review_date fields).
3. Acquisition site/delivery meanings of Q-code and b/t suffixes need confirmation before declaring site-disjoint performance.

**Outcome:** v1.7 is independently executable and has passed this evidence check. The dataset audit code can be frozen. This does **not** certify the exported labels as correct, the dHash candidates as duplicates, or the November split protocol as finalized.
