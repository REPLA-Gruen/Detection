$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
$AppName="CCBRE_AI_LAB_Context_Foreign_Material_v1_2"
python -m pip install pyinstaller
if($LASTEXITCODE -ne 0){throw "PyInstaller installation failed"}
python -m PyInstaller --noconfirm --clean --onedir --windowed --name $AppName --icon ".\assets\CCBRE_AI_LAB_Logo.ico" --add-data ".\assets\CCBRE_AI_LAB_Logo.png;assets" --add-data ".\assets\CCBRE_AI_LAB_Logo.ico;assets" --add-data ".\config.json;." --collect-all ultralytics --collect-all torch --collect-all torchvision --hidden-import PIL._tkinter_finder .\STANDALONE_APP.py
if($LASTEXITCODE -ne 0){throw "Windows app build failed"}
$Dist=Join-Path $PSScriptRoot "dist\$AppName"
Copy-Item ".\config.json" $Dist -Force
Write-Host "Built: $Dist" -ForegroundColor Green
