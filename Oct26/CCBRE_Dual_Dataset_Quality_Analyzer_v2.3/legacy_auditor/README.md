**CURRENT STANDALONE EXECUTION:** See `README_STANDALONE.md` before anything below. The following text is retained as historical v1.6 methods, not required v1.7 inputs.

**v1.7 UPDATE:** Read `README_V17.md` FIRST for July annotation reconciliation, required `-ReferenceLabels` and `-ReferenceYaml`, day-level grouping, and no-decode postprocessing. The following text retains the historical v1.6 instructions.

# CCBRE | October 2026 raw dataset auditor v1.6

**Read-only evidence-generation package. No Word report is generated.** This updates v1.5 in response to review findings. It has passed synthetic tests in the development environment; **the complete 9,628-image October dataset and the original July source photographs are not available here, and have not been executed or reverified for v1.6.** A Word report must be revised only after the real results are inspected.

## Inputs and prerequisites

The full October audit reads *only* these two exports within `C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw`:

- `biowaste-collection-images/{images,labels,labels-detect}`
- `biowaste-floorspread-images/{images,labels,labels-detect}`

Both `data.yaml`, `report.md`, and `.sync-state.json` are reconciled separately. **Raw is never edited.** Outputs must be outside `Raw`.

Requires **Python 3.10+** (`int.bit_count()`), Pillow, PyYAML and Matplotlib. The user's Python 3.13 / Pillow 12.3 combination is supported by the standard dependency calls, but actual full-run performance and decoder fingerprints must be recorded from the user's computer.

```powershell
py -3 -m pip install -r requirements.txt
py -3 .\test_smoke.py
py -3 .\test_v14_features.py
py -3 .\test_v15_regression.py
py -3 .\test_v16_regression.py
```

## Recommended execution order

**A. July reference fingerprinting.** First determine the location of the ORIGINAL **July source image** directory or directories (not the recursively nested dataset builds, crops, testing copies, or experiment outputs). This is a manual source-selection decision. For example:

```powershell
.\RUN_JULY_FINGERPRINT.ps1 -JulyImageFolders @('C:\Hadi\Foreign_Material\Data\Ours\Jul-26\<ACTUAL ORIGINAL SOURCE IMAGES DIRECTORY>')
```

Alternatively, explicitly supply one or more `--images-root` arguments:

```powershell
py -3 .\analyze_oct26.py --fingerprint-only --images-root "C:\...\Jul-26\...\images" --output "C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.6"
```

This mode scans only **directly contained** image files by default. Use `--recursive`/`-Recursive` only when the selected original source image directory has genuine nested originals. It **does not check** that these are original, unique July images. Check `31_fingerprint_runtime.json` and the count against the authoritative 3,359-image July source corpus. For multiple image directories, use multiple `--images-root` flags. Image labeling/metadata is NOT needed in fingerprint-only mode.

**B. October complete audit, with July comparison.**

```powershell
.\RUN_ANALYSIS.ps1 -ReferenceManifest "C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.6\31_fingerprint_manifest.csv"
```

Or do a complete October run without reference and later rerun with the reference. Direct Python CLI:

```powershell
py -3 .\analyze_oct26.py --dataset "C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw" --output "C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.6" --reference-manifest "C:\...\Jul-26\Original_Source_Fingerprints_v1.6\31_fingerprint_manifest.csv"
```

Output defaults to `Oct-26\Analysis_v1.6`. **Do not use `--no-sha256` in the evidence run.** Reruns with an unchanged dataset, reference and algorithm should be consistent, but the new dHash does not have universal perceptual invariance. Preserve the entire output folder.

## Improvements over v1.5

### 1. Distinct images versus source copies

The global dHash search now retains **one representative per decoded EXIF-display pixel SHA-256** (or a byte hash fallback if decoding is unavailable). Exact-byte or pixel-identical copies therefore **do not multiply near-duplicate candidate rows**. `23_dhash_candidates.csv` includes the representative pair and the **complete original image memberships**, membership counts, both source contexts, candidate group keys and cross-group/context indicators.

`08_exact_duplicates.csv` and `28_pixel_content_duplicates.csv` continue to separately document confirmed byte-identical and pixel-identical groups. A pixel-identical copy with altered metadata is not a new independent image. A lossy re-encode can require dHash screening and review.

### 2. Distribution for screening-threshold selection

`29_dhash_distance_distribution.csv` counts representative-pair Hamming distances **0–64** (the report displays 0–12 and the number >12), calculated over all unique-pixel pairs. `23_dhash_candidates.csv` saves candidate pairs at **distance <=12**; `selected_for_extended` marks those at the current `--dhash-threshold` (default **6**). This makes the threshold inspectable without relying on a single arbitrary hit count. On very homogeneous images a single dHash distance may not separate real duplicates from unrelated green-waste textures. A candidate remains **unverified**. Output candidate CSV rows are capped at two million, with an explicit `review_rows_truncated` flag in `01_overview.json`; full numeric histograms and extended connected-component statistics are still computed.

### 3. Camera-specific and timestamp-specific floorspread runs

`file_group_key` now derives from `(EXIF Make, EXIF Model, contiguous DSC number)`, with missing camera identity assigned to an explicit unknown-camera bucket. `time_gap_group_key` separately records sessions that break at >10-minute capture-time gaps, reversals, or breaks in DSC sequence. Groups inferred without enough EXIF data are **provisional**, and camera *model* alone does not prove camera-*body* identity. Both keys and counts are included in 02/21/24 and 01_overview.json. The time-gap grouping is an additional diagnostic, not silently substituted into the strict group rule. Parse failures produce explicit issues.

### 4. July–October overlap (new optional two-pass procedure)

`--fingerprint-only` creates `31_fingerprint_manifest.csv`, `31_fingerprint_runtime.json` and `31_fingerprint_quality_issues.csv` from **explicit original-source image directories**. Supply this manifest through `--reference-manifest` during the October run:

- `32_reference_strong_matches.csv`: October-to-July byte-identical matches and (if decoder compatibility is confirmed) decoded-pixel-identical matches.
- `33_reference_dhash_candidates.csv`: unverified approximate matches, one pair per unique decoded-pixel group, with member paths and distances up to 12.
- `34_reference_overlap_summary.json`: reference image counts, strong overlaps, candidates, runtime compatibility and limitations; also embedded in the October overview.

File SHA-256 comparisons are portable across environments. **Pixel and dHash comparisons are permitted only if the July reference runtime JSON matches Pillow version, JPEG codec version and signature algorithms.** If the runtime differs or JSON is missing, the analyzer conservatively skips the weaker comparisons. This does not establish capture or site independence: physically different files can still depict the same delivery.

### 5. Unified October grouping for November splitting

A union-find algorithm now assigns **one ID per October image**:

- `split_group_strict` = provisional filename grouping + SHA-256 exact-file groups + decoded-pixel duplicate groups.
- `split_group_extended` = strict groups plus dHash candidate edges at current threshold.

`31_split_group_manifest.csv` has one record per image with both IDs and group sizes. `35_split_group_size_distribution.csv` and `36_split_group_summary.csv` summarize connected components. **The extended IDs are strictly advisory**: single false-positive dHash edges can chain large numbers of otherwise unrelated samples. Do not use extended IDs as a binding split constraint until visually adjudicated. Strict group IDs are also *provisional* because Q-code meaning, b/t suffix semantics, and DSC session boundaries need partner confirmation. The grouping currently covers October only; the July reference outputs supply links for the subsequent combined Nov-26 builder, not automatically merged July/October IDs.

### 6. Orientation and decode robustness

A failed EXIF-transpose operation attempts a raw encoded-image decode for hashing and logs `exif_transpose_failed_encoded_fallback`; `pixel_coordinate_frame` records this distinction so such pixels are not silently asserted to be EXIF-display pixels. Image header dimensions remain available if full decode fails. Sheets for all positive EXIF-rotated images continue to compare two label-coordinate hypotheses. Reviewer sheet filenames contain a sanitized original image stem followed by a short SHA-256 token instead of only a hash.

## Source inventory and scientific interpretation

The audit still exports all earlier v1.5 reports, especially image and object manifests, all 29 classes, detection/polygon consistency, exporter source and class reconciliation, SHA duplicate annotation conflict tiers, source label ignore/rename discrepancies, per-view-letter statistics, and short-side pixel dimensions at 1536 px. `05_resolution_statistics.csv` uses **EXIF-display dimensions**, also retaining encoded width/height. Once full v1.6 results are validated, the Word report §4.1 should be updated to present display dimensions as primary and encoded dimensions in an appendix; the old report is not silently updated using unverified v1.6 values.

**Interpretation gates**: (1) check all sources and metadata reconcile; (2) inspect error records and orientation sheets; (3) visually adjudicate dHash candidates and annotation conflicts; (4) confirm Q-code and b/t letter semantics and staged floorspread origin with the partner; (5) validate the exact original July source root and all cross-version duplicate links. A zero exact overlap does not prove independent acquisition or deliveries. No model is trained or evaluated by this audit.

## Outputs introduced in v1.6

| File | Purpose |
|---|---|
| `23_dhash_candidates.csv` | Unique-pixel representative pairs, all members, review distances <=12 |
| `29_dhash_distance_distribution.csv` | 65 Hamming counts for distances 0–64, aggregated over representatives |
| `21_capture_group_manifest.csv` | Camera-aware run and time-gap keys alongside existing provenance |
| `24_capture_group_summary.csv` | Per-group camera and timestamp coverage |
| `31_fingerprint_manifest.csv` | July fingerprint-only mode only |
| `31_fingerprint_runtime.json` | Environment signature for July fingerprint-only mode only |
| `32_reference_strong_matches.csv` | Cross-version file/pixel identity matches if reference provided |
| `33_reference_dhash_candidates.csv` | Cross-version approximate matches if reference provided |
| `34_reference_overlap_summary.json` | Cross-version overlap accounting if reference provided |
| `31_split_group_manifest.csv` | October one-image/one-row strict and extended grouping IDs |
| `35_split_group_size_distribution.csv` | Group-size distribution for detecting giant folds |
| `36_split_group_summary.csv` | Full group membership and group-level diagnostics |
| `01_overview.json` | New run counts, EXIF parseability, reference match status, group sizes, dHash histogram |

All other prior output filenames remain valid. Save the entire `Analysis_v1.6` directory and (if used) the separate July fingerprint folder for independent review.

### Post-run threshold sensitivity without repeating image decoding

After examining `29_dhash_distance_distribution.csv` and candidate examples at distances 0–12, you may rebuild **advisory** grouping IDs at a different threshold from existing results:

```powershell
py -3 .\rebuild_split_groups.py --analysis "C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.6" --dhash-threshold 8
```

This creates a **new** `Analysis_v1.6\SplitGroups_dhash08\` directory containing `31_split_group_manifest.csv`, `35_split_group_size_distribution.csv`, `36_split_group_summary.csv`, and `rebuild_summary.json`. No image re-decoding is needed and existing results remain unchanged. It **refuses** to produce incomplete groups if the original dHash candidate CSV was truncated at the two-million-row safety cap. It cannot recover pairs beyond Hamming distance 12; a full new scan is required for those cutoffs.
