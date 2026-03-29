@echo off
setlocal enabledelayedexpansion
title ApplyPilot - TEST: Apply Dry Run (5 jobs)
color 0E

echo.
echo  ================================================================
echo    TEST 2: Apply DRY RUN (5 jobs)
echo    Forms filled but NOT submitted. Safe to run.
echo    Requires: Claude Code CLI installed + account active
echo    Started: %date% %time%
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"

cd /d "%PROJECT%"
call .venv\Scripts\activate.bat

:: ── Pre-flight checks ─────────────────────────────────────────────
echo  [1/3] Pre-flight checks...

:: Check Claude Code CLI
where claude >nul 2>&1
if %errorlevel% neq 0 (
    echo        [ERROR] Claude Code CLI not found on PATH!
    echo        Install from: https://claude.ai/code
    pause
    exit /b 1
)
echo        Claude Code CLI: OK

:: Check ready jobs
for /f %%i in ('"%PYTHON%" -c "from applypilot.database import get_connection, init_db; init_db(); c=get_connection(); print(c.execute('SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL').fetchone()[0])"') do set "READY=%%i"
echo        Ready to apply: %READY% jobs
if "%READY%"=="0" (
    echo        [ERROR] No tailored resumes ready! Run TEST_PIPELINE.bat first.
    pause
    exit /b 1
)
echo.

:: ── Clear stale locks ─────────────────────────────────────────────
echo  [2/3] Clearing stale locks...
"%PYTHON%" _clear_locks.py 2>nul
echo        Done.
echo.

:: ── Dry-run apply (5 jobs) ────────────────────────────────────────
echo  ================================================================
echo  [3/3] DRY RUN: Applying to 5 jobs (forms filled, NOT submitted)
echo        Model: sonnet  |  Workers: 1  |  Headless: no
echo  ================================================================
echo.
"%PYTHON%" -m applypilot apply --dry-run --limit 5
echo.

echo  ================================================================
echo    Dry run complete! %date% %time%
echo    Check logs: %USERPROFILE%\.applypilot\logs\
echo.
echo    Happy with results? Run TEST_APPLY_LIVE.bat for real submissions.
echo  ================================================================
pause
