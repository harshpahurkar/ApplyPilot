@echo off
setlocal enabledelayedexpansion
title ApplyPilot - Full Pipeline + Apply
color 0A

echo.
echo  ================================================================
echo    ApplyPilot - Full Pipeline + Continuous Apply
echo    Discover ^> Enrich ^> Score ^> Tailor ^> Cover ^> PDF ^> Apply
echo    Started: %date% %time%
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"
set "LOG_DIR=%USERPROFILE%\.applypilot\logs"
set "WORKERS=4"
set "MIN_SCORE=6"
set "APPLY_LIMIT=630"
set "MODEL=haiku"

cd /d "%PROJECT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
call .venv\Scripts\activate.bat

:: ── Clear stale locks ─────────────────────────────────────────────
echo  [1/4] Clearing stale locks...
"%PYTHON%" _clear_locks.py 2>nul
echo        Done.
echo.

:: ── Pipeline ──────────────────────────────────────────────────────
echo  ================================================================
echo  [2/4] Running pipeline (streaming, %WORKERS% workers)
echo  ================================================================
echo.
"%PYTHON%" -m applypilot run --stream -w %WORKERS% --min-score %MIN_SCORE%
if %errorlevel% neq 0 (
    echo.
    echo  [WARNING] Pipeline had errors. Continuing to apply...
    echo.
)

:: ── Status ────────────────────────────────────────────────────────
echo.
echo  [3/4] Pipeline status:
echo  ================================================================
"%PYTHON%" -m applypilot status
echo.

:: ── Apply ─────────────────────────────────────────────────────────
echo  ================================================================
echo  [4/4] Starting auto-apply (%APPLY_LIMIT% jobs, model=%MODEL%, headless)
echo        Press Ctrl+C once  = skip current job
echo        Press Ctrl+C twice = stop everything
echo  ================================================================
echo.
"%PYTHON%" -m applypilot apply --limit %APPLY_LIMIT% --model %MODEL% --headless --min-score %MIN_SCORE%
echo.

echo  ================================================================
echo    All done! %date% %time%
echo    Results: %USERPROFILE%\.applypilot\
echo    Logs:    %LOG_DIR%\
echo  ================================================================
pause
