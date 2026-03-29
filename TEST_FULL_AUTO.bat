@echo off
setlocal enabledelayedexpansion
title ApplyPilot - FULL AUTO (Pipeline + Apply, all at once)
color 0A

echo.
echo  ================================================================
echo    FULL AUTO MODE
echo    Pipeline + Apply running simultaneously
echo    Discover ^> Enrich ^> Score ^> Tailor ^> Cover ^> PDF ^> APPLY
echo    Started: %date% %time%
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"

cd /d "%PROJECT%"
call .venv\Scripts\activate.bat

:: ── Pre-flight checks ─────────────────────────────────────────────
echo  [1/3] Pre-flight checks...

where claude >nul 2>&1
if %errorlevel% neq 0 (
    echo        [ERROR] Claude Code CLI not found on PATH!
    pause
    exit /b 1
)
echo        Claude Code CLI: OK
echo.

echo  [WARNING] This will discover jobs AND submit REAL applications.
echo            Pipeline stages run in parallel with apply.
echo            2 Chrome workers, threshold 30 ready jobs.
echo.
set /p "CONFIRM=Type YES to continue: "
if /i not "%CONFIRM%"=="YES" (
    echo  Aborted.
    pause
    exit /b 0
)
echo.

:: ── Clear stale locks ─────────────────────────────────────────────
echo  [2/3] Clearing stale locks...
"%PYTHON%" _clear_locks.py 2>nul
echo        Done.
echo.

:: ── Full auto ─────────────────────────────────────────────────────
echo  ================================================================
echo  [3/3] FULL AUTO: Pipeline + Apply (concurrent)
echo        Pipeline workers: 4  |  Apply workers: 2
echo        Apply threshold:  30 ready jobs
echo        Model: sonnet     |  Min score: 6
echo        Press Ctrl+C to stop everything
echo  ================================================================
echo.
"%PYTHON%" -m applypilot auto -w 4 -a 2 --threshold 30 --min-score 6
echo.

echo  ================================================================
echo    Full auto complete! %date% %time%
echo    Results: %USERPROFILE%\.applypilot\
echo    Logs:    %USERPROFILE%\.applypilot\logs\
echo  ================================================================
pause
