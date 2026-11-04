# CCBRE Dual-Dataset Quality Analyzer v2.3 — Standalone + Offline HTML Case Browsers

**Purpose:** start from the original July and October photographs, labels, and metadata. Produce (i) the complete numerical audit, (ii) general statistical figures and tables, (iii) photo examples and complete, searchable lists for every issue, (iv) editable Swiss-provider review lists and (v) a full offline HTML report. **No prior v1.6/v1.7 results or software installation are required.** The tested v1.7.1 analysis code is bundled in `legacy_auditor/`.

## 1. Requirements and installation (Windows)

Python 3.10+ on Windows. Inside this extracted package folder, open PowerShell and run:

```powershell
py -3 -m pip install -r requirements.txt
.\RUN_TESTS.ps1
```

The tests use small synthetic photographs. They are not a substitute for the real dataset audit.

## 2. Input directories

The defaults reflect the existing workstation structure:

```
C:\Hadi\Foreign_Material\Data\Ours\Jul-26\images\       # 3,359 original photographs ONLY
C:\Hadi\Foreign_Material\Data\Ours\Jul-26\labels\       # July YOLO labels; recursive tree accepted
C:\Hadi\Foreign_Material\Data\Ours\Jul-26\dataset.yaml  # original July class map
C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw\          # October S3 dataset, as downloaded
```

The October Raw folder needs both `biowaste-collection-images` and `biowaste-floorspread-images`, each containing `images`, `labels`, `labels-detect`, `data.yaml`, `report.md` and `.sync-state.json`.

**Do not use July training copies, tiles or crops** instead of the 3,359 authoritative source photographs. Do not put the output inside either raw-data directory.

## 3. One command for all analysis and full HTML

```powershell
.\RUN_ALL.ps1
```

By default this creates:

```
C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.3
```

To specify custom paths or sample sizes:

```powershell
.\RUN_ALL.ps1 `
  -JulyImages 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\images' `
  -JulyLabels 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\labels' `
  -JulyYaml   'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\dataset.yaml' `
  -OctoberRaw 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw' `
  -Output     'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.3' `
  -SamplesPerProblem 30
```

`RUN_ALL.ps1` normally generates visual sheets for **all substantive duplicate-image disagreements**, and the selected examples and complete CSV review queues for other problems. It creates July fingerprints *itself*, without relying on older results. It does not change original photographs or labels. The program deliberately refuses to overwrite existing numerical results; choose a new output directory for another raw dataset snapshot.

## 4. What to open and send

**Open `Dual_Dataset_Quality_v2.3\OPEN_FIRST.html` in Chrome/Edge.**

It is a complete locally navigable HTML review. After **A. The problem**, click **View all N cases** to open a searchable, paginated offline table of every matching candidate case. If a real photograph comparison has already been rendered, that row also has an **Open photo** link. The report only embeds the selected photograph examples; the case list itself is complete. This avoids copying all 4,841 empty-labeled originals or other thousands of images into the report ZIP.

The main report and the individual pages have a fixed presentation for all nine problems:

- A. The problem
- B. Photographic/numerical evidence
- C. Important observations
- D. Actions required from Swiss partners and CCBRE

The HTML uses images inside the **same output folder**, so do not send the `.html` alone. Also provided:

- `GENERAL_STATISTICS.html`: all general plots and tables (19 charts, 16 CSV tables in `00_General_Dataset_Analysis/General_Statistics/` for the current reference export).
- `PROBLEM_01.html` through `PROBLEM_09.html`: issue-by-issue evidence galleries, linked records and review forms.
- `ALL_CASES_01.html` through `ALL_CASES_09.html`: searchable, paginated complete case lists linked by prominent buttons directly after each problem description. The lists include EVERY row in the defined candidate population, not only illustrated examples. Optional supporting full lists have their own pages.
- `ALL_CASES_MANIFEST.json`: exact counts and definitions for every all-cases button. The original evidence files are preserved. The new all-case CSVs are derived browsable indexes.
- `CORRECTIVE_ACTIONS.html`: meeting requests and blank decision log.
- `EVIDENCE_INDEX.html`: clickable references to the complete evidence package.
- `99_Technical_Verification/`: complete underlying audit, raw-label matching, run metadata, file checks and July fingerprints.

**ZIP the entire `Dual_Dataset_Quality_v2.3` results directory and send it for the Word report.** Do not include the original 54 GiB of images: selected evidence images and linked CSV data are saved inside the analysis output.

## 5. Regenerate charts and HTML without decoding the source photographs

For a completed earlier dual-dataset result directory, including `Dual_Dataset_Quality_v2.0`, run:

```powershell
.\RUN_REPORT_ONLY.ps1 -Analysis 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.0'
```

This recreates the general statistics and HTML pages from **that directory's own audit**. It does not rerun or reverify the original image/annotation files. To regenerate only the HTML when the statistics already exist:

```powershell
.\RUN_HTML_ONLY.ps1 -Analysis 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.3'
```

## 6. Scientific interpretation and checks

- A file with no object annotations is **not proof of a clean load**.
- Camera-recorded dates are **not independently verified physical collection dates**. The Swiss photography team must review discrepancies.
- The class names in July and October are different. July context boxes `Container` / `Wagon` are counted separately from foreign-material objects.
- Exact duplicates must be distinguished from merely similar green-waste scenes.
- The 30 orientation comparisons are **AI-assisted suggestions awaiting dated human sign-off**.
- The report presents observed evidence and requested decisions. It does **not** measure detector performance or approve any disputed annotation version.
- Frozen 2026 totals appear only as comparisons in `Frozen_Baseline_Comparison.csv`. If the inputs change, differences are reported, **not silently corrected**.

For a final partner document, share the results ZIP together with any completed human review/annotation decisions. The Word report should be produced from the saved evidence and should not invent missing provider answers.

## 7. Fast upgrade of the already completed v2.2 results (recommended)

There is **no need to rerun July fingerprinting, image analysis or the v2.2 program** merely to add the buttons. From the extracted v2.3 source folder:

```powershell
.\RUN_HTML_ONLY.ps1 -Analysis 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.2'
```

The generator creates searchable `ALL_CASES_XX.html` pages, plus their CSV indexes, and refreshes `OPEN_FIRST.html`, the nine problem pages and the other HTML navigation. Original audit results, charts and images remain unchanged.

**Case-list definitions:** 135 duplicate pairs with differing annotation signatures (107 substantive), 554 July-photo candidates with fewer boxes in at least one October annotation version (536 in every version), 22 images containing only unmapped July categories, 4,841 empty-labeled image records, 349 non-default EXIF-oriented records (211 have two-way sheets), 63 collection Q-code date-review groups, 3,229 collection stop groups, 21 rare categories and nine corrective-action topics. Supporting pages include all 221 duplicate pairs, 9,628 camera-metadata rows, 211 orientation sheets, 29 category summaries and other relevant tables. These are review candidates, not independently confirmed errors.

**When sharing:** the v2.3 HTML update ZIP contains only the HTML pages and case CSVs. Extract it **inside the existing v2.2 output folder**, not beside it. Image links refer to files already in that folder. For colleagues without the earlier evidence package, share the complete v2.2 results ZIP after upgrading.
