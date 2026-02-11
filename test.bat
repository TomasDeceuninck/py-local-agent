@echo off
CALL .\.venv\Scripts\activate
.\.venv\Scripts\python.exe evaluations/evaluate_agent.py
PAUSE