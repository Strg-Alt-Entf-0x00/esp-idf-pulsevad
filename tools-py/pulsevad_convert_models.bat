@echo off
setlocal
echo =======================================================
echo   PulseVAD ESP32-P4 - Model Converter
echo =======================================================
echo.
echo Converting PyTorch models in models/py/ to ESPM and C-Header...
echo.

cd /d "%~dp0"

echo [1/1] Exporting PulseVAD Models (Header + ESPM)...
python pulsevad_models_converter\pulsevad_convert_models.py
if errorlevel 1 (
    echo.
    echo [ERROR] PulseVAD model conversion failed.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo Done! Models are saved in:
echo   - src/pulsevad_model_data.h
echo   - models/espm/
echo =======================================================
pause
exit /b 0
