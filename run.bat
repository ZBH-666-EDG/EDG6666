@echo off
title 跌倒监测系统

set PYTHON=E:\python\python.exe

echo Checking Python...
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found at %PYTHON%
    echo Please edit run.bat and set PYTHON to your python.exe path
    pause
    exit /b 1
)

cd /d "%~dp0"
echo Working dir: %CD%
echo.

echo Installing dependencies...
"%PYTHON%" -m pip install -r requirements.txt -q 2>nul

echo Starting server at http://localhost:5001
echo Close this window to stop
echo ========================================
echo.

start http://localhost:5001
"%PYTHON%" app.py

pause
