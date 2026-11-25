@echo off
rem SmishGuard launcher for Windows. Needs Python 3.12 or newer on PATH.
rem The first run creates .venv and installs requirements.txt (needs internet), then starts the app.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.12 or newer from python.org and tick "Add python.exe to PATH".
        pause
        exit /b 1
    )
    echo Creating the virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import streamlit, sklearn, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Installing the requirements. This takes a few minutes the first time ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

echo Starting SmishGuard at http://localhost:8501 . Close this window to stop it.
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:failed
echo Setup failed. See the messages above.
pause
exit /b 1
