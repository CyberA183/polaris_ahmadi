@echo off
echo Building Polaris as a standalone executable...
echo This may take several minutes...

REM Install/update all runtime dependencies from requirements.txt
echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Create the executable
python -m PyInstaller --clean --noconfirm polaris.spec

echo.
echo Build complete! Check the 'dist' folder for the executable.
echo You can copy the built Polaris app from the 'dist' folder.
pause