@echo off
cls
title Hrudhi Personal Assistant - AI-Powered Note Taking
color 0D

echo.
echo =============================================
echo   🤖 Hrudhi Personal Assistant 🤖
echo =============================================
echo   Advanced AI-powered note taking with chat
echo   Modern Qwen2.5-7B model integrated
echo   Learning capabilities with links support
echo =============================================
echo.

cd /d "%~dp0"

echo 🤖 Starting Hrudhi Personal Assistant...
echo 📚 This includes note-taking + AI chat in one app
echo ⏳ First time: May download ~7GB AI model
echo 💬 Then you can chat with AI about your notes!
echo 🧠 Teach AI new information and share links
echo.

REM Try to use virtual environment if available
if exist ".venv\Scripts\python.exe" (
    echo 🔧 Using virtual environment...
    .venv\Scripts\python.exe hrudhi_personal_assistant.py
) else (
    echo 🔧 Using system Python...
    python hrudhi_personal_assistant.py
)

if %errorlevel% equ 0 (
    echo.
    echo ✅ Application closed successfully!
) else (
    echo.
    echo ❌ Error occurred. Make sure you have:
    echo   - Python installed
    echo   - Required dependencies (run: pip install -r requirements.txt)
    echo   - At least 8GB free space for AI models
    echo   - Internet connection (first AI model download)
    echo.
    pause
)

timeout /t 2 >nul