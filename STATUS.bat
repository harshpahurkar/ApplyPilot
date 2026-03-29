@echo off
title ApplyPilot - Status Log
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python _status.py
echo.
pause
