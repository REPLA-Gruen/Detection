param(
    [string]$Analysis = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.2'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3 .\general_statistics_v22.py --analysis $Analysis
if ($LASTEXITCODE -ne 0) { throw 'General figures/tables update failed. Original results remain unchanged.' }
Write-Host "Additional charts and tables: $Analysis\00_General_Dataset_Analysis\General_Statistics" -ForegroundColor Green
