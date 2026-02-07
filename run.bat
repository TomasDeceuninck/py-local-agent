@echo off
REM This script activates the virtual environment and starts the Python agent.

echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo Starting the agent...
python main.py
