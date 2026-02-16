@echo off
title RoboTrader Launcher

echo ================================
echo   Starting RoboTrader Platform
echo ================================

REM -------- BACKEND (FastAPI) --------
start "RoboTrader Backend" cmd /k "I: & cd /d I:\RoboTrader\backend & I:\RoboTrader\backend\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

REM -------- FRONTEND (Next.js) --------
start "RoboTrader Frontend" cmd /k "I: & cd /d I:\RoboTrader\webapp & npm run dev -- -H 0.0.0.0"

REM -------- WAIT --------
timeout /t 10 >nul

REM -------- OPEN BROWSER --------
start http://localhost:3000
start http://127.0.0.1:8000/docs

echo ================================
echo   RoboTrader Started
echo ================================
pause
