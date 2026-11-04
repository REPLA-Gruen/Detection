# Release notes — CCBRE Dual-Dataset Quality Analyzer v2.2

- Complete **offline HTML**: `OPEN_FIRST.html` is the full review, with nine consistent problems, examples, numerical evidence, notes and required actions.
- Links to individual `PROBLEM_01.html` ... `PROBLEM_09.html` pages with full review photo galleries, `GENERAL_STATISTICS.html`, `CORRECTIVE_ACTIONS.html` and `EVIDENCE_INDEX.html`.
- Complete general-statistics suite: 19 PNG figures and 16 CSV tables, including all-class image support, source record counts and log-area **bin fractions**, not misleading linear-density normalisation.
- Independently runs July fingerprints, original July label inventory, full October inventory, duplicate analysis and July/October reconciliation using bundled tested v1.7.1 engine.
- Problem evidence includes duplicate label versions, missing July markings, taxonomy changes, empty labels, orientation, dates, context views, representativeness and provider decision lists.
- Previous general-statistics extension required the historical 3,359/9,628 snapshot and failed on new/synthetic datasets. v2.2 uses input-derived counts, verifies internal consistency and reports differences from the historical snapshot separately.
- `RUN_REPORT_ONLY.ps1` re-renders charts and HTML from a **completed analysis output** without decoding original photographs. This provides a fast upgrade path for users who already ran v2.0.
- All files run locally. No internet requests, CDN scripts, web servers or image-generation substitutions. Raw photos and original labels remain unchanged.

Verification: all seven synthetic/regression test scripts passed. Complete HTML tested on the existing 9,628-image v2.0 **result manifests** and 107 comparison sheets, with 1,310 local links checked and none broken. The v2.2 full source-data audit is to be run by the user on the original files.
