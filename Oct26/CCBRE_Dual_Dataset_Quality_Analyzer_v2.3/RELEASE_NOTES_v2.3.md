# CCBRE Dual-Dataset Quality Analyzer v2.3 — all-case browsers

Only the **HTML/report presentation layer** changes compared with v2.2. All core July and October statistics, image comparisons, data-quality tests and material-count calculations are unchanged. The `quality_evidence.py`, `general_statistics_v22.py`, and bundled `legacy_auditor` source remain unmodified.

**New capabilities**
- A prominent **View all N cases** button immediately after subsection A (The problem) for each of the nine issues, on both OPEN_FIRST and individual problem pages.
- Searchable/paginated `ALL_CASES_01.html` ... `ALL_CASES_09.html` with full downloadable derived CSV indexes; supporting views include all rows in the original source tables.
- Rendered photo links for rows where the v2.2 run already produced a comparison sheet. All other cases are still listed with their original identifiers.
- Case-population rules explicitly identified (not a silent choice between earliest/latest label versions); human findings and approval fields stay blank.
- `ALL_CASES_MANIFEST.json` records the case definitions and row counts.

**Fast upgrade of the already-generated full v2.2 evidence**
```powershell
.\RUN_HTML_ONLY.ps1 -Analysis "C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.2"
```
No image decoding, July hashing or new full analysis is necessary.

**Verification performed here:** end-to-end synthetic v2.2/v2.3 integration passed; focused HTML case-browser test passed; report-only v2.3 applied successfully to the complete uploaded v2.2 evidence with 9/9 buttons and all case tables. HTML assets/links were checked for existence (none missing) and three sample browser JavaScript programs passed Node's syntax check.
