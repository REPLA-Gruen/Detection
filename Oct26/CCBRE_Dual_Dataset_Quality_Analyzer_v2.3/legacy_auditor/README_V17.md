# CCBRE Oct-26 raw-data auditor v1.7 — July label reconciliation

**Scientific status:** v1.7 extends the verified v1.6 auditor. It does not change the source dataset and does not train a model. On full October images, it adds capture-day groups and audits the *original July annotations*, including context exclusion and one-to-one class-aware IoU comparison with the October export. It deliberately does **not** choose a canonical annotation from conflicting October duplicates.

**Python requirements:** Python 3.10+, Pillow, PyYAML, matplotlib. Run all five tests before an evidence run.

```powershell
py -3 -m pip install -r requirements.txt
py -3 .\test_smoke.py
py -3 .\test_v14_features.py
py -3 .\test_v15_regression.py
py -3 .\test_v16_regression.py
py -3 .\test_v17_regression.py
```

## Inputs

1. October folder containing the two exports `biowaste-collection-images` and `biowaste-floorspread-images` (each with `images/`, `labels/`, `labels-detect/`, `data.yaml`, `report.md`, `.sync-state.json`).
2. **July source-image fingerprint manifest**: `31_fingerprint_manifest.csv`, which v1.7 can generate itself using `RUN_JULY_FINGERPRINT.ps1`. An existing v1.6-generated manifest can be reused, but it is optional.
3. **July YOLO label directory** (`-ReferenceLabels`). This can be the parent folder containing `train/labels`, `val/labels`, and `test/labels`, searched recursively. Do NOT point to a larger workspace with many incompatible training derivatives.
4. **July data.yaml** (`-ReferenceYaml`). Must correspond to the July labels' numeric class IDs, not the October 29-class schema.
5. Optional manual class-map CSV (`-ReferenceClassMap`), exactly two columns `july_class,october_class`. `Diverses` is kept unmapped under the executed baseline. Hypothetical heterogeneous classes such as `Other` must also remain unmapped if encountered. `Styropor/Dämmung` remains unmapped unless the explicit `Insulation` override is supplied.

### Full evidence run

Replace `<PATH_TO_JULY_LABELS>` and `<PATH_TO_JULY_DATA_YAML>` with actual local paths, and use the **v1.7-generated** July reference manifest path (see `README_STANDALONE.md`):

```powershell
.\RUN_ANALYSIS.ps1 `
  -ReferenceManifest "C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7\31_fingerprint_manifest.csv" `
  -ReferenceLabels "<PATH_TO_JULY_LABELS>" `
  -ReferenceYaml "<PATH_TO_JULY_DATA_YAML>"
```

Outputs: `C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7` by default. The `Raw` input stays unchanged.

If the July fingerprint manifest is missing, recreate it using the existing `RUN_JULY_FINGERPRINT.ps1` against **July original source images only**, not derived crops/training augmentations. Set `-ReferenceManifest` to the new file.

### Second pass without image decoding

If the first run finds additional unmapped July classes, produce a *two-column* class mapping like this (example only; do not accept this particular equivalence without checking class semantics):

```csv
july_class,october_class
Styropor/Dämmung,Insulation
```

Then run:

```powershell
.\RUN_LABEL_DIFF_ONLY.ps1 `
  -AnalysisInput "C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7" `
  -ReferenceManifest "C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7\31_fingerprint_manifest.csv" `
  -ReferenceLabels "<PATH_TO_JULY_LABELS>" `
  -ReferenceYaml "<PATH_TO_JULY_DATA_YAML>" `
  -ReferenceClassMap "<PATH_TO_TWO_COLUMN_MAP.csv>"
```

A new `LabelDiff_SecondPass` directory stores the revised diff **and day-level grouping outputs (37/38)**. No October image is decoded again and original evidence is preserved. You may use an existing validated `Analysis_v1.6` folder as `-AnalysisInput` for this fast postprocessing, without repeating its image analysis.

## New analysis outputs

- `37_day_level_split_groups.csv` — one October image per row with `split_group_day`. Container/wagon images grouped by `(Q-code, stop)` and floorspread images grouped by **EXIF calendar capture day**. Same-file and equal-decoded-pixel hashes connect groups. Missing floorspread EXIF date falls back to a camera/numeric sequence group, explicitly marked.
- `38_day_level_group_members.csv`, `38_day_level_group_summary.json` — complete members, size distribution, and capture-day totals. On the independently verified v1.6 manifest, the expected check is **3,236 groups**, largest **920**. If the full run differs, inspect paths, timestamps, and hash/group rules before using a split.
- `39_reference_label_diff_summary.json` — per-source counts, YOLO object rows (including context), mapped and unmapped July classes, and an interval of same-class IoU≥0.50 matches across *all* matching October label versions.
- `40_reference_label_diff_per_image.csv` — original July image, label path/status, exclusion and mapped counts, October version count, minimum/maximum matches, flag for divergent October GT.
- `41_reference_class_mapping.csv` — July-to-October names/IDs, mapping decisions and provenance; the observed `Diverses` remains unmapped.
- `42_reference_label_object_matches.csv` — matched pairs with IoU and evidence paths (each October candidate version separately).
- `43_reference_label_diff_issues.csv` — missing labels, multiple nonidentical label files with the same stem, parser errors; these images are not silently counted as clean.
- `44_reference_label_comparison_by_oct_image.csv` — version-by-version comparison, no silent newer-is-better assumption.
- `45_reference_class_diff_counts.csv` — July class statistics and bounded matched counts per class.

### Important data interpretation

The historical report's **5,640** is the count of *foreign-material annotations* used for the July model build. The raw July YOLO source labels may **additionally include Container and Wagon scene annotations**. v1.7 computes the actual source total, separates context rows, then compares foreign-material rows to 5,640. It will not assume 5,640 includes context annotations.

The audited July source taxonomy contains no `Other` category. If another reference ontology contains a heterogeneous `Other` category it should remain unmapped. `Diverses` remains unresolved because the exporter lists it under both ignored and renamed labels. All auto mappings are explicit in `41_reference_class_mapping.csv` and can be reviewed before interpreting the overlap metric.

Byte-identical July-to-October images can have **multiple different October annotation versions**. v1.7 evaluates every version and reports the minimum/maximum across them. It never silently picks the later timestamp. Historical labels may themselves have been revised, and IoU matching alone does not validate material identity.

The capture-day split is safer than sequence-gap grouping for floorspread, but **a day is an operational proxy for a session**, not independently certified site identity. Keep all frames from the same day in the same fold, pending partner acquisition records.

## Scope limits and expected first checks

First compare `38_day_level_group_summary.json` with 3,236/920. Then check `39_reference_label_diff_summary.json` and `43_reference_label_diff_issues.csv`. The July total can only be reconciled once the supplied July labels and `data.yaml` have been read. The existing 29-class October statistics and all image-level EXIF, orientation sheets and overlays remain generated by the main analyzer. G3 human sign-off is still a separate record, unrelated to label diff.
