param(
    [Parameter(Mandatory=$true)][string]$ReferenceManifest,
    [Parameter(Mandatory=$true)][string]$ReferenceLabels,
    [Parameter(Mandatory=$true)][string]$ReferenceYaml,
    [string]$ReferenceClassMap = '',
    [string]$AnalysisInput = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7',
    [string]$Output = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7\LabelDiff_SecondPass'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3 -c 'import yaml, PIL, matplotlib'
if ($LASTEXITCODE -ne 0) { throw 'Install required packages: py -3 -m pip install -r requirements.txt' }
$Argv = @('.\analyze_oct26.py','--label-diff-only','--analysis-input',$AnalysisInput,
          '--output',$Output,'--reference-manifest',$ReferenceManifest,
          '--reference-labels',$ReferenceLabels,'--reference-yaml',$ReferenceYaml)
if ($ReferenceClassMap -ne '') { $Argv += @('--reference-class-map',$ReferenceClassMap) }
py -3 @Argv
if ($LASTEXITCODE -ne 0) { throw 'Label-diff second pass failed.' }
Write-Host "Label-diff outputs: $Output" -ForegroundColor Green
