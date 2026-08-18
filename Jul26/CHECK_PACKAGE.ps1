$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
python .\30_verify_package_code.py
if($LASTEXITCODE -ne 0){throw "Package code verification failed"}
