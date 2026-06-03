$ErrorActionPreference = "Stop"
Write-Host "Setting up Python 3.12 local environment..."

# 1. Create directory python_env
$envDir = "$PSScriptRoot\python_env"
if (-not (Test-Path $envDir)) {
    New-Item -ItemType Directory -Force -Path $envDir | Out-Null
}

# 2. Download Python 3.12 embeddable
$zipPath = "$envDir\python-3.12.8.zip"
Write-Host "Downloading Python 3.12.8 Embeddable..."
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip" -OutFile $zipPath

# 3. Extract Python 3.12
Write-Host "Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $envDir -Force
Remove-Item $zipPath

# 4. Configure ._pth to enable site-packages
Write-Host "Configuring Python path..."
$pthFile = "$envDir\python312._pth"
$pthContent = @"
python312.zip
.

# Enable site packages
import site
"@
Set-Content -Path $pthFile -Value $pthContent

# 5. Download get-pip.py
$getPipPath = "$envDir\get-pip.py"
Write-Host "Downloading pip installer..."
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath

# 6. Install pip
Write-Host "Installing pip..."
Start-Process -FilePath "$envDir\python.exe" -ArgumentList "$getPipPath" -Wait -NoNewWindow
Remove-Item $getPipPath

# 7. Install requirements
Write-Host "Installing requirements (tensorflow, transformers, pandas, scikit-learn, tqdm)..."
Start-Process -FilePath "$envDir\Scripts\pip.exe" -ArgumentList "install tensorflow transformers pandas scikit-learn tqdm" -Wait -NoNewWindow

Write-Host "Python 3.12 and TensorFlow are successfully set up locally!"
