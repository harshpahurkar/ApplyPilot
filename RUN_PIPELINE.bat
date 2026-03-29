@echo off
setlocal enabledelayedexpansion
title ApplyPilot - Pipeline Only (No Apply)
color 0B

echo.
echo  ================================================================
echo    ApplyPilot - Pipeline Only
echo    Discover ^> Enrich ^> Score ^> Tailor ^> Cover ^> PDF
echo    (No auto-apply — just prepare everything)
echo    Started: %date% %time%
echo  ================================================================
echo.

set "PROJECT=C:\Users\Harsh\Desktop\Other Cool Projects\ApplyPilot"
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"

cd /d "%PROJECT%"
call .venv\Scripts\activate.bat

echo  [1/2] Running full pipeline (streaming, 4 workers)...
echo.
"%PYTHON%" -m applypilot run --stream -w 4 --min-score 6
echo.

echo  [2/2] Final status:
echo  ================================================================
"%PYTHON%" -m applypilot status
echo.

echo  ================================================================
echo    Pipeline complete! %date% %time%
echo    Tailored resumes: %USERPROFILE%\.applypilot\tailored_resumes\
echo    Cover letters:    %USERPROFILE%\.applypilot\cover_letters\
echo.
echo    To apply: double-click RUN.bat
echo  ================================================================
pause
