# CCBRE October Raw Auditor v1.7 — Standalone execution (corrected)

## Does v1.7 need v1.6?
**No.** The scripts and Python modules in this ZIP are sufficient to create the July fingerprint manifest and then audit October and reconcile July labels. A prior `Original_Source_Fingerprints_v1.6/31_fingerprint_manifest.csv` can be reused to save time, but is an *optional existing input artifact*, not a dependency on v1.6 code or output. This corrected guide generates all required artifacts using v1.7.

## Inputs you still must supply
- Original July **source images** (one or more folders, no training/validation derivatives or crop copies).
- Original July YOLO **labels** folder(s) and matching July `data.yaml` / `dataset.yaml` with July class IDs.
- Original October `Raw` directory, containing both exporter folders and metadata.
- Python 3.10+ with packages in `requirements.txt`.

## Run from PowerShell in the unzipped v1.7 code folder

```powershell
py -3 -m pip install -r requirements.txt
py -3 .\test_smoke.py
py -3 .\test_v14_features.py
py -3 .\test_v15_regression.py
py -3 .\test_v16_regression.py
py -3 .\test_v17_regression.py
```

### Step 1 — Create a *v1.7* July fingerprint manifest
Replace the placeholder path with the folder holding only the original 3,359 July images. If the originals reside in multiple source directories, supply all of them in the array.

```powershell
.\RUN_JULY_FINGERPRINT.ps1 `
  -JulyImageFolders @('C:\ACTUAL\PATH\TO\JULY\ORIGINAL\images') `
  -Output 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7'
```

Add `-Recursive` only if the source images really live in nested folders. Check `31_fingerprint_manifest.csv` and `31_fingerprint_quality_issues.csv`. Expect **3,359 unique original July photographs** if you have selected the authoritative complete original corpus (not derived training views). Investigate unexpected counts or hashing failures before continuing.

### Step 2 — Full standalone October audit + July comparison
Set the correct July labels root and its *July* YAML file. The label root may contain nested `train/labels`, `val/labels`, and `test/labels` directories; the tool searches recursively.

```powershell
.\RUN_ANALYSIS.ps1 `
  -Dataset 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw' `
  -Output 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7' `
  -ReferenceManifest 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7\31_fingerprint_manifest.csv' `
  -ReferenceLabels 'C:\ACTUAL\PATH\TO\JULY\labels' `
  -ReferenceYaml 'C:\ACTUAL\PATH\TO\JULY\dataset.yaml'
```

This runs the v1.7 image audit, image overlap, July label comparison, day grouping, and output generation. It does **not** require the v1.6 code or any `Analysis_v1.6` directory.

### Step 3 — Verify the evidence outputs
- `01_overview.json`: image totals, EXIF and audit status.
- `34_reference_overlap_summary.json`: July/October image-content match.
- `38_day_level_group_summary.json`: group counts; earlier post-hoc evidence suggests 3,236 groups, maximum 920, for the same data and grouping assumptions.
- `39_reference_label_diff_summary.json`: label-count reconciliation.
- `43_reference_label_diff_issues.csv`: unexpected missing/ambiguous July labels.

Use `RUN_LABEL_DIFF_ONLY.ps1` to rerun the class-name mapping against the new `Analysis_v1.7` folder without re-decoding, if necessary. Follow `README_V17.md` for the `-ReferenceClassMap` CSV format.

## Important provenance distinction
Earlier post-hoc `LabelDiff_From_v1.6` outputs are **not** proof that the full standalone v1.7 execution has been run. Preserve their original executed script separately when available. The steps above create a *new* integrated v1.7 evidence record with its own provenance. Archiving v1.6 remains valuable for historical reproducibility, but it is not necessary to execute v1.7.
