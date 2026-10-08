@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Preparing OloHeart for its first run...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
        echo.
        echo OloHeart setup could not be completed.
        pause
        exit /b 1
    )
)

"%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
set "oloheart_exit_code=%ERRORLEVEL%"

if not "%oloheart_exit_code%"=="0" (
    echo.
    echo OloHeart closed with error code %oloheart_exit_code%.
    pause
)

exit /b %oloheart_exit_code%
