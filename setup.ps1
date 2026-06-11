$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvPath = Join-Path $Root ".conda-env"
$CachePath = Join-Path $Root "cache"
$ModelsPath = Join-Path $Root "models"

New-Item -ItemType Directory -Force -Path $CachePath | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsPath | Out-Null

$env:PYTHONUTF8 = "1"
$env:TORCH_HOME = $ModelsPath
$env:XDG_CACHE_HOME = $CachePath
$env:NUMBA_CACHE_DIR = Join-Path $CachePath "numba"
$env:PIP_CACHE_DIR = Join-Path $CachePath "pip"
$env:DEMUCS_CACHE = $ModelsPath
$env:HF_HOME = Join-Path $ModelsPath "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ModelsPath "huggingface\hub"
$env:HF_HUB_DISABLE_XET = "1"

if (Test-Path $EnvPath) {
    Write-Host "Conda environment already exists: $EnvPath"
    Write-Host "Updating base tools..."
    conda install -y -p $EnvPath -c conda-forge python=3.11 ffmpeg
} else {
    Write-Host "Creating local conda environment: $EnvPath"
    conda create -y -p $EnvPath -c conda-forge python=3.11 ffmpeg
}

Write-Host "Installing Python packages..."
$PythonExe = Join-Path $EnvPath "python.exe"
$env:PATH = "$EnvPath;$EnvPath\Library\bin;$EnvPath\Scripts;$env:PATH"
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install demucs torchcodec faster-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "Downloading Demucs model weights..."
& $PythonExe "$Root\download_demucs_models.py"

Write-Host "Preloading Demucs models..."
& $PythonExe "$Root\preload_models.py"

Write-Host "Preloading Whisper speech recognition model..."
& $PythonExe "$Root\preload_whisper_models.py"

Write-Host ""
Write-Host "Done. Start the tool with Run.bat"
Write-Host "Project folder: $Root"
