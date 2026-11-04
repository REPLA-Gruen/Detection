param(
    [string]$JulyImages = 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\images',
    [string]$JulyLabels = 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\labels',
    [string]$JulyYaml = 'C:\Hadi\Foreign_Material\Data\Ours\Jul-26\dataset.yaml',
    [string]$OctoberRaw = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Raw',
    [string]$Output = 'C:\Hadi\Foreign_Material\Data\Ours\Oct-26\Dual_Dataset_Quality_v2.3',
    [string]$ClassMap = '',
    [ValidateRange(1,200)][int]$SamplesPerProblem = 20,
    [switch]$AllDuplicateSheets = $true,
    [switch]$EvidenceOnly
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3 -c 'import yaml, PIL, matplotlib'
if ($LASTEXITCODE -ne 0) { throw 'Missing dependencies. Run: py -3 -m pip install -r requirements.txt' }
$Cmd = @('.\run_from_scratch.py', '--july-images', $JulyImages, '--july-labels', $JulyLabels,
        '--july-yaml', $JulyYaml, '--october-root', $OctoberRaw, '--output', $Output,
        '--samples-per-problem', "$SamplesPerProblem")
if ($ClassMap) { $Cmd += @('--reference-class-map',$ClassMap) }
if ($AllDuplicateSheets) { $Cmd += '--all-duplicate-sheets' }
if ($EvidenceOnly) { $Cmd += '--evidence-only' }
py -3 @Cmd
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed. Do NOT use partial outputs as final evidence.' }
Write-Host "Finished. Open full HTML report: $Output\OPEN_FIRST.html" -ForegroundColor Green
