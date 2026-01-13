@echo off
echo Setting up ASEP Project...
echo.

echo Creating directories...
if not exist "uploads\images" mkdir uploads\images
if not exist "uploads\videos" mkdir uploads\videos
if not exist "uploads\audio" mkdir uploads\audio
if not exist "plates" mkdir plates
echo Directories created!
echo.

echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment already exists.
)
echo.

echo Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo.

echo Setup complete!
echo.
echo To run the application:
echo   1. venv\Scripts\activate
echo   2. python app.py
echo.
pause
