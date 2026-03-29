@echo off
setlocal enabledelayedexpansion
title ApplyPilot - TEST: Pipeline Only
color 0B

echo.
echo  ================================================================
echo    TEST 1: Pipeline Only (no apply)
echo    Discover ^> Enrich ^> Score ^> Tailor ^> Cover ^> PDF
echo    Started: %date% %time%
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"

cd /d "%PROJECT%"
call .venv\Scripts\activate.bat

:: ── Clear stale locks ─────────────────────────────────────────────
echo  [1/3] Clearing stale locks...
"%PYTHON%" _clear_locks.py 2>nul
echo        Done.
echo.

:: ── Pipeline (streaming mode, 4 workers) ──────────────────────────
echo  [2/3] Running pipeline (streaming, 4 workers)...
echo        All stages run concurrently - DB is the conveyor belt.
echo.
"%PYTHON%" -m applypilot run --stream -w 4 --min-score 6
echo.

:: ── Status ────────────────────────────────────────────────────────
echo  [3/3] Final status:
echo  ================================================================
"%PYTHON%" -m applypilot status
echo.

echo  ================================================================
echo    Pipeline test complete! %date% %time%
echo.
echo    If jobs are ready, run TEST_APPLY_DRY.bat next.
echo  ================================================================
pause
