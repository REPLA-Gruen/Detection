param(
  [string]$Analysis = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.2'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3 .\html_report.py --analysis $Analysis
if ($LASTEXITCODE -ne 0) { throw 'HTML generation failed; inspect the console message.' }
Write-Host "Open: $Analysis\OPEN_FIRST.html" -ForegroundColor Green
