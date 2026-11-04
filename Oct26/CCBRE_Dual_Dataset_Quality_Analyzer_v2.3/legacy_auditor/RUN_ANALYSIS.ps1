param(
    [string]$Dataset = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw',
    [string]$Output = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Analysis_v1.7',
    [string]$ReferenceManifest = '',
    [string]$ReferenceLabels = '',
    [string]$ReferenceYaml = '',
    [string]$ReferenceClassMap = '',
    [ValidateRange(0,12)][int]$DhashThreshold = 6
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3 -c 'import yaml, PIL, matplotlib'
if ($LASTEXITCODE -ne 0) { throw 'Install required packages: py -3 -m pip install -r requirements.txt' }
if (($ReferenceLabels -ne '') -xor ($ReferenceYaml -ne '')) { throw 'Pass both -ReferenceLabels and -ReferenceYaml.' }
if (($ReferenceLabels -ne '') -and ($ReferenceManifest -eq '')) { throw 'Pass -ReferenceManifest together with July labels.' }
$Argv = @('.\analyze_oct26.py','--dataset',$Dataset,'--output',$Output,'--dhash-threshold',"$DhashThreshold")
if ($ReferenceManifest -ne '') { $Argv += @('--reference-manifest',$ReferenceManifest) }
if ($ReferenceLabels -ne '') { $Argv += @('--reference-labels',$ReferenceLabels) }
if ($ReferenceYaml -ne '') { $Argv += @('--reference-yaml',$ReferenceYaml) }
if ($ReferenceClassMap -ne '') { $Argv += @('--reference-class-map',$ReferenceClassMap) }
Write-Host 'CCBRE October v1.7 | read-only raw data and July comparison | full dataset decode.'
py -3 @Argv
if ($LASTEXITCODE -ne 0) { throw 'v1.7 analysis failed; inspect console and 43_reference_label_diff_issues.csv.' }
Write-Host "Analysis completed: $Output" -ForegroundColor Green
