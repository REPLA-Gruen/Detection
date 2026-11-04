# Test and verification record — v2.2

All tests below passed in the development environment (Python 3.13, Pillow available):

1. `legacy_auditor/test_smoke.py` — PASS
2. `legacy_auditor/test_v14_features.py` — PASS
3. `legacy_auditor/test_v15_regression.py` — PASS
4. `legacy_auditor/test_v16_regression.py` — PASS
5. `legacy_auditor/test_v17_regression.py` — PASS
6. `legacy_auditor/test_overlay_display_frame.py` — PASS
7. `test_v22_integration.py` — PASS (fresh July + October tiny synthetic input, all nine evidence folders, charts, offline HTML).

The HTML reporter and revised chart producer were also tested on the **user-provided v2.0 analysis outputs**. The regenerated charts and all 14 report HTML pages rendered from those manifests with zero missing local image/link paths. This is an evidence/report integration check, **not a new full 9,628-image run of v2.2**.

The real full run must still be conducted on the user's Windows source directories.
