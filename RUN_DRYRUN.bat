@echo off
setlocal enabledelayedexpansion
title ApplyPilot - DRY RUN (No Real Submissions)
color 0E

echo.
echo  ================================================================
echo    ApplyPilot - DRY RUN
echo    Full pipeline + apply (forms filled but NOT submitted)
echo    Started: %date% %time%
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"

cd /d "%PROJECT%"
call .venv\Scripts\activate.bat

:: ── Clear locks ───────────────────────────────────────────────────
echo  [1/3] Clearing stale locks...
"%PYTHON%" _clear_locks.py 2>nul
echo        Done.
echo.

:: ── Pipeline ──────────────────────────────────────────────────────
echo  [2/3] Running pipeline (streaming, 4 workers)...
echo.
"%PYTHON%" -m applypilot run --stream -w 4 --min-score 6
echo.

:: ── Dry-run apply (5 jobs) ────────────────────────────────────────
echo  ================================================================
echo  [3/3] DRY RUN apply (5 jobs, forms filled but NOT submitted)
echo  ================================================================
echo.
"%PYTHON%" -m applypilot apply --dry-run --limit 5
echo.

echo  ================================================================
echo    Dry run complete! Review results before running RUN.bat
echo  ================================================================
pause
