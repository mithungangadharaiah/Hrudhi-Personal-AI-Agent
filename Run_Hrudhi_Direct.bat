@echo off
echo.
echo ========================================
echo    🤖 Hrudhi Personal AI Agent 🤖
echo    ✨ Fancy UI with Robotic Face ✨ 
echo ========================================
echo.
echo Starting Hrudhi with creative interface...
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo Using virtual environment...
    ".venv\Scripts\python.exe" main.py
) else (
    echo Using system Python...
    python main.py
)

echo.
echo 🤖 Hrudhi has closed. Press any key to exit...
pause > nul