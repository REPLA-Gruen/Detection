param(
    [string]$Analysis = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7',
    [switch]$ExpectedSnapshot,
    [string]$OutputJSON = ''
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Args = @('.\verify_secondary_v17.py','--analysis',$Analysis)
if ($ExpectedSnapshot) { $Args += '--expected-snapshot' }
if ($OutputJSON) { $Args += @('--output-json',$OutputJSON) }
py -3 @Args
if ($LASTEXITCODE -ne 0) { throw 'Secondary dataset reconciliation FAILED.' }
Write-Host 'Secondary reconciliation checks passed.' -ForegroundColor Green
