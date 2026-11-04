param(
    [Parameter(Mandatory=$true)][string[]]$JulyImageFolders,
    [string]$Output = 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\Original_Source_Fingerprints_v1.7',
    [switch]$Recursive
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3 -c 'import yaml, PIL, matplotlib'
if ($LASTEXITCODE -ne 0) { throw 'Missing dependencies: py -3 -m pip install -r requirements.txt' }
$Arguments = @('.\analyze_oct26.py','--fingerprint-only','--output',$Output)
foreach ($Folder in $JulyImageFolders) { $Arguments += @('--images-root', $Folder) }
if ($Recursive) { $Arguments += '--recursive' }
Write-Host 'Fingerprinting ONLY the July image directories you selected. Avoid training/derived image trees.'
py -3 @Arguments
if ($LASTEXITCODE -ne 0) { throw 'July fingerprinting failed.' }
Write-Host ('July fingerprint manifest: ' + (Join-Path $Output '31_fingerprint_manifest.csv')) -ForegroundColor Green
