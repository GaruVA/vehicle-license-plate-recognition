@echo off
REM install_system.bat
REM Vehicle License Plate Recognition (VLPR) System Installation Script for Windows

SETLOCAL ENABLEDELAYEDEXPANSION

REM Get absolute path to the script directory
SET "SCRIPT_DIR=%~dp0"
CD /D "%SCRIPT_DIR%"

ECHO Setting up Vehicle License Plate Recognition (VLPR) System
ECHO ========================================================

REM Check for Python installation
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO [ERROR] Python is not installed. Please install Python 3.8+ and rerun this script.
    EXIT /B 1
)

REM Create virtual environment
IF EXIST venv_vlpr (
    ECHO [WARNING] Virtual environment already exists. Removing old one...
    rmdir /S /Q venv_vlpr
)
python -m venv venv_vlpr

REM Activate virtual environment
CALL "%SCRIPT_DIR%venv_vlpr\Scripts\activate"

REM Upgrade pip
python -m pip install --upgrade pip

REM Try to install CUDA-enabled PyTorch (CUDA 12.9)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
IF ERRORLEVEL 1 (
    ECHO [WARNING] CUDA-enabled PyTorch install failed, falling back to CPU-only version...
    python -m pip install torch torchvision torchaudio
)

python -m pip install opencv-python
python -m pip install ultralytics numpy Pillow onnxruntime easyocr flask requests pytest black flake8

ECHO [INFO] Python packages installed successfully!

REM System verification
IF NOT EXIST "%SCRIPT_DIR%system_verification.py" (
    ECHO [ERROR] system_verification.py not found in %SCRIPT_DIR%
    EXIT /B 1
)
python "%SCRIPT_DIR%system_verification.py"
IF ERRORLEVEL 1 (
    ECHO [ERROR] Package verification failed!
    EXIT /B 1
)

ECHO Installation completed successfully!
ECHO ========================================================
ECHO Vehicle License Plate Recognition System Ready!
ECHO ========================================================
ECHO Next Steps:
ECHO 1. Activate the environment: CALL venv_vlpr\Scripts\activate
ECHO 2. Verify the setup: python system_verification.py
ECHO 3. Run detection: python main.py
ECHO Important Files:
ECHO - main.py                   - Main detection script
ECHO - system_verification.py    - System verification script
ECHO - models\                   - YOLO model files directory
ECHO - tests\                    - Test video and image files
ECHO For more information, visit:
ECHO   https://github.com/GaruVA/vehicle-license-plate-recognition
ECHO Remember to:
ECHO - Place your trained models in the models\ directory
ECHO - Update camera URLs in the scripts as needed
ECHO - Ensure proper lighting for optimal detection
ECHO Happy detecting!
ENDLOCAL