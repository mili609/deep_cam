@echo off
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" run.py --execution-provider cuda
