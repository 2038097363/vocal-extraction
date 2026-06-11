@echo off
setlocal
cd /d "%~dp0"
set "TORCH_HOME=%~dp0models"
set "XDG_CACHE_HOME=%~dp0cache"
set "NUMBA_CACHE_DIR=%~dp0cache\numba"
set "PIP_CACHE_DIR=%~dp0cache\pip"
set "DEMUCS_CACHE=%~dp0models"
set "HF_HOME=%~dp0models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%~dp0models\huggingface\hub"
set "HF_HUB_DISABLE_XET=1"
set "PYTHONUTF8=1"
set "PATH=%~dp0.conda-env;%~dp0.conda-env\Library\bin;%~dp0.conda-env\Scripts;%PATH%"
"%~dp0.conda-env\pythonw.exe" "%~dp0voice_extractor.py"
