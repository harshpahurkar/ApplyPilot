@echo off
setlocal enabledelayedexpansion
title ApplyPilot - Full Auto Run
color 0A

echo.
echo  ================================================================
echo    ApplyPilot - Full Auto Run
echo    Pipeline + Apply (continuous, fully automated)
echo    Started: %date% %time%
echo  ================================================================
echo.
echo    Workers:       4 (pipeline) + 1 (apply browser)
echo    Min Score:     6
echo    Threshold:     3 (apply starts after 3 jobs ready)
echo    Headless:      No (Chrome visible)
echo    CAPTCHA:       CapSolver enabled
echo    LLM:           DeepSeek
echo    Apply:         Claude Code (Pro)
echo.
echo    Press Ctrl+C once  = skip current job
echo    Press Ctrl+C twice = stop everything
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"
set "LOG_DIR=%USERPROFILE%\.applypilot\logs"

cd /d "%PROJECT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
call .venv\Scripts\activate.bat

:: ── Clear stale locks ─────────────────────────────────────────────
echo  [1/3] Clearing stale locks...
"%PYTHON%" _clear_locks.py 2>nul
echo        Done.
echo.

:: ── Status before run ─────────────────────────────────────────────
echo  [2/3] Current queue status:
echo  ================================================================
"%PYTHON%" -m applypilot status
echo.

:: ── Full auto run ─────────────────────────────────────────────────
echo  ================================================================
echo  [3/3] Starting full auto run (pipeline + apply)
echo        %date% %time%
echo  ================================================================
echo.
"%PYTHON%" -m applypilot auto --workers 4 --apply-workers 1 --min-score 6 --threshold 3
echo.

echo  ================================================================
echo    All done! %date% %time%
echo    Results: %USERPROFILE%\.applypilot\
echo    Logs:    %LOG_DIR%\
echo  ================================================================
pause
