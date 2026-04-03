@echo off
setlocal enabledelayedexpansion
title ApplyPilot - Stealth Auto Apply (Sonnet + Sonnet)
color 0B

echo.
echo  ================================================================
echo    ApplyPilot - Stealth Auto Apply
echo    Claude Code (Sonnet) for apply, Copilot (Sonnet) for tailoring
echo    Started: %date% %time%
echo  ================================================================
echo.
echo    Apply model:   claude-sonnet-4.6 via copilot-api proxy
echo    Tailor model:  claude-sonnet-4.6 via Copilot CLI
echo    Workers:       4 pipeline + 4 apply
echo    Pacing:        Stealth ON for Copilot scoring/tailoring
echo    Job timeout:   420s (7 min)
echo    Mode:          Full auto (discover+score+tailor+apply)
echo.
echo    Press Ctrl+C once  = skip current job
echo    Press Ctrl+C twice = stop everything
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"
set "LOG_DIR=%USERPROFILE%\.applypilot\logs"
set "PROXY_PORT=4141"

:: Pipeline settings
set "PIPELINE_WORKERS=4"
set "APPLY_WORKERS=4"
set "MIN_SCORE=6"
set "APPLY_THRESHOLD=3"
set "APPLY_MODEL=claude-sonnet-4.6"
set "APPLY_BACKEND=claude"

cd /d "%PROJECT%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
call .venv\Scripts\activate.bat

:: ── Stealth pacing config (for Copilot CLI scoring/tailoring) ─────
set APPLYPILOT_STEALTH_ENABLED=1
set APPLYPILOT_STEALTH_JOB_DELAY_MIN=15
set APPLYPILOT_STEALTH_JOB_DELAY_MAX=45
set APPLYPILOT_STEALTH_LLM_DELAY_MIN=2
set APPLYPILOT_STEALTH_LLM_DELAY_MAX=5
set APPLYPILOT_STEALTH_WARMUP_EXTRA=5
set APPLYPILOT_STEALTH_ROTATE=0

:: Budget: 80 requests/hour with soft throttle at 55
set APPLYPILOT_STEALTH_BUDGET_WINDOW=3600
set APPLYPILOT_STEALTH_BUDGET_MAX=80
set APPLYPILOT_STEALTH_BUDGET_SOFT=55

:: ── Copilot LLM (scoring/tailoring) ──────────────────────────────
set APPLYPILOT_USE_COPILOT_LLM=1
set COPILOT_LLM_MODEL=claude-sonnet-4.6
set APPLYPILOT_COPILOT_LLM_MIN_INTERVAL=5
set APPLYPILOT_COPILOT_LLM_RATE_LIMIT_COOLDOWN=300
set APPLYPILOT_COPILOT_LLM_MAX_WORKERS=4

:: Chat slots
set APPLYPILOT_COPILOT_GLOBAL_CHAT_SLOTS=5
set APPLYPILOT_COPILOT_TAILORING_CHAT_SLOTS=4
set APPLYPILOT_COPILOT_SCORING_CHAT_SLOTS=2
set APPLYPILOT_COPILOT_LLM_SERIALIZE=0

:: ── Apply settings (Claude Code via copilot-api proxy) ────────────
set APPLYPILOT_JOB_TIMEOUT=420

:: ── Step 1: Start copilot-api proxy ──────────────────────────────
echo  [1/4] Starting copilot-api proxy on port %PROXY_PORT%...

:: Kill any existing proxy
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PROXY_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>nul
)

start "copilot-api-proxy" /MIN cmd /c "npx -y copilot-api@latest start --port %PROXY_PORT% > "%LOG_DIR%\proxy.log" 2>&1"

:: Wait for proxy
echo        Waiting for proxy...
set "PROXY_READY=0"
for /L %%i in (1,1,30) do (
    if !PROXY_READY!==0 (
        powershell -c "try { (Invoke-WebRequest -Uri 'http://localhost:%PROXY_PORT%/v1/models' -TimeoutSec 2 -UseBasicParsing).StatusCode } catch { exit 1 }" >nul 2>nul
        if !errorlevel!==0 (
            set "PROXY_READY=1"
            echo        Proxy is up!
        ) else (
            timeout /t 2 /nobreak >nul
        )
    )
)
if !PROXY_READY!==0 (
    echo  [WARNING] Proxy may not be ready yet. Continuing anyway...
)
echo.

:: ── Step 2: Clear stale locks ─────────────────────────────────────
echo  [2/4] Clearing stale locks...
"%PYTHON%" -c "from applypilot.database import get_connection; db=get_connection(); db.execute('UPDATE jobs SET apply_status=NULL WHERE apply_status=''in_progress'''); db.commit(); print('       Done.')"
echo.

:: ── Step 3: Status ────────────────────────────────────────────────
echo  [3/4] Current queue status:
echo  ================================================================
"%PYTHON%" -m applypilot status
echo.

:: ── Step 4: Full auto run (pipeline + apply) ──────────────────────
echo  ================================================================
echo  [4/4] Starting full auto (pipeline + apply)
echo        %date% %time%
echo  ================================================================
echo.
"%PYTHON%" -m applypilot auto ^
    --workers %PIPELINE_WORKERS% ^
    --apply-workers %APPLY_WORKERS% ^
    --min-score %MIN_SCORE% ^
    --threshold %APPLY_THRESHOLD% ^
    --model %APPLY_MODEL% ^
    --agent-backend %APPLY_BACKEND% ^
    --stealth
echo.

echo  ================================================================
echo    Finished! %date% %time%
echo    Results: %USERPROFILE%\.applypilot\
echo    Logs:    %LOG_DIR%\
echo  ================================================================

:: ── Cleanup: kill proxy ───────────────────────────────────────────
echo  Shutting down proxy...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PROXY_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>nul
)
echo  Proxy stopped.
pause
