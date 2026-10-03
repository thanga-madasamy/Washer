@echo off
cd /d "%~dp0"
python washer_poc.py %*
if errorlevel 1 pause
