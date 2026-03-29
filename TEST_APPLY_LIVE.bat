@echo off
setlocal enabledelayedexpansion
title ApplyPilot - TEST: Apply LIVE (10 jobs)
color 0C

echo.
echo  ================================================================
echo    TEST 3: Apply LIVE (10 real submissions!)
echo    THIS WILL ACTUALLY SUBMIT APPLICATIONS.
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

where claude >nul 2>&1
if %errorlevel% neq 0 (
    echo        [ERROR] Claude Code CLI not found on PATH!
    pause
    exit /b 1
)
echo        Claude Code CLI: OK

for /f %%i in ('"%PYTHON%" -c "from applypilot.database import get_connection, init_db; init_db(); c=get_connection(); print(c.execute('SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL').fetchone()[0])"') do set "READY=%%i"
echo        Ready to apply: %READY% jobs
if "%READY%"=="0" (
    echo        [ERROR] No jobs ready! Run TEST_PIPELINE.bat first.
    pause
    exit /b 1
)

echo.
echo  [WARNING] This will submit REAL applications to REAL companies.
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

:: ── Live apply (10 jobs) ──────────────────────────────────────────
echo  ================================================================
echo  [3/3] LIVE: Applying to 10 jobs
echo        Model: sonnet  |  Workers: 1
echo        Press Ctrl+C to skip current job, Ctrl+C x2 to stop
echo  ================================================================
echo.
"%PYTHON%" -m applypilot apply --limit 10
echo.

echo  ================================================================
echo    Live test complete! %date% %time%
echo    Check results: %USERPROFILE%\.applypilot\logs\
echo.
echo    Ready for full run? Use TEST_FULL_AUTO.bat
echo  ================================================================
pause
