@echo off
title RoboTrader Agentic AI
echo ========================================
echo   Starting RoboTrader Agentic AI Module
echo ========================================

REM Using the main backend venv for simplicity, but in a separate process
I:
cd /d I:\RoboTrader\agentic_trading
I:\RoboTrader\backend\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8001

pause
