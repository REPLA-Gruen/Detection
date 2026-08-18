param(
    [ValidateSet("check","create-target","verify-target","create-context","verify-context","train-context","import-context","evaluate-context","train-foreign-base","import-phase4b-base","import-foreign","prepare-uavvaste","verify-uavvaste","finetune-stagea","compare-stagea","consolidate","verify-final","compare-foreign","test-complete","app","all-existing","all-from-data")]
    [string]$Stage = "check",
    [switch]$Overwrite,
    [switch]$Download
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ArgsList=@(".\run_pipeline.py","--stage",$Stage)
if($Overwrite){$ArgsList+="--overwrite"}
if($Download){$ArgsList+="--download"}
python @ArgsList
if($LASTEXITCODE -ne 0){throw "Final Context+Hybrid pipeline failed with exit code $LASTEXITCODE"}
