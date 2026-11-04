$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$tests = @('test_v22_integration.py','test_v23_browser.py',
  'legacy_auditor/test_smoke.py',
  'legacy_auditor/test_v14_features.py',
  'legacy_auditor/test_v15_regression.py',
  'legacy_auditor/test_v16_regression.py',
  'legacy_auditor/test_v17_regression.py',
  'legacy_auditor/test_overlay_display_frame.py')
foreach ($test in $tests) {
  Write-Host "TEST: $test" -ForegroundColor Cyan
  py -3 $test
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $test" }
}
Write-Host 'All available synthetic/regression tests passed.' -ForegroundColor Green
