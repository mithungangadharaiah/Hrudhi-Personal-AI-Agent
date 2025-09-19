@echo off
cls
title Hrudhi Personal Assistant
color 0B

echo.
echo =============================================
echo   🤖 Hrudhi Personal Assistant 🤖
echo =============================================
echo.

cd /d "%~dp0"

echo � Starting your AI Personal Assistant...
echo ⏳ Loading AI models (first time takes 30 seconds)...
echo.

python hrudhi_personal_assistant.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Thank you for using Hrudhi!
) else (
    echo.
    echo ❌ Error occurred. Check that Python is installed.
    pause
)

timeout /t 2 >nul