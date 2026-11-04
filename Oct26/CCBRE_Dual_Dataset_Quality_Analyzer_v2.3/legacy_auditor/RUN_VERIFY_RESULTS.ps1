param(
    [string]$Analysis = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7',
    [switch]$ExpectedSnapshot
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Args = @('.\verify_v17_results.py','--analysis',$Analysis)
if ($ExpectedSnapshot) { $Args += '--expected-snapshot' }
py -3 @Args
if ($LASTEXITCODE -ne 0) { throw 'The independent v1.7 evidence checks FAILED. Review the diagnostics before using the report.' }
Write-Host 'Independent evidence checks passed.' -ForegroundColor Green
