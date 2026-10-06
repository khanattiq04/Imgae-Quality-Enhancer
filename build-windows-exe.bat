@echo off
setlocal

py -3 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/windows/
  echo During installation, select "Add python.exe to PATH", then run this file again.
  pause
  exit /b 1
)

py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements-desktop.txt
py -3 -m PyInstaller --noconfirm --clean --onefile --windowed --name ScreenshotHDPrintEnhancer --add-data "public;public" --collect-all webview --collect-all cv2 desktop.py

echo.
echo Done. Your installer-free desktop app is here:
echo dist\ScreenshotHDPrintEnhancer.exe
pause
