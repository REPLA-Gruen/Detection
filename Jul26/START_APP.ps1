$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
python .\STANDALONE_APP.py --config .\config.json
if($LASTEXITCODE -ne 0){throw "Standalone app failed with exit code $LASTEXITCODE"}
