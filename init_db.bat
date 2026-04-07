@echo off
cd /d "%~dp0"
python init_db.py
if %ERRORLEVEL% neq 0 pause
