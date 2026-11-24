@echo off
setlocal
echo =======================================================
echo   PulseVAD Upstream Models Downloader
echo =======================================================
echo.
set "PROJECT_ROOT=%~dp0.."
pushd "%PROJECT_ROOT%" || exit /b 1
python "%~dp0pulsevad_models_py_downloader\pulsevad_download_models_py.py" --project-root "%PROJECT_ROOT%" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Model download failed (Exit Code: %EXIT_CODE%).
    pause
    exit /b %EXIT_CODE%
)
echo.
echo [OK] PulseVAD models are downloaded and verified in models/py/
pause
exit /b 0
