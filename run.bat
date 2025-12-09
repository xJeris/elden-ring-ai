@echo off
REM Quick launcher for Elden Ring AI with Python 3.12
REM Menu system for behavioral cloning and training options

cls
echo ================================================================
echo           ELDEN RING AI - LAUNCHER
echo ================================================================
echo.
echo Choose an option:
echo.
echo   [1] Train AI with Your Recorded Gameplay
echo   [2] Record New Gameplay (Basic - Actions Only)
echo   [3] Record with Camera Tracking (Recommended!)
echo   [4] Train AI from Scratch (no recording)
echo   [5] Run Custom Python Script
echo.
echo   [0] Exit
echo.
echo ================================================================

set /p choice="Enter your choice (0-5): "

if "%choice%"=="0" goto end
if "%choice%"=="1" goto train
if "%choice%"=="2" goto record
if "%choice%"=="3" goto record_camera
if "%choice%"=="4" goto clone
if "%choice%"=="5" goto custom

REM Invalid choice
cls
echo.
echo Invalid choice. Exiting...
echo.
timeout /t 2 >nul
goto end

:train
cls
echo Starting AI Training with Your Recorded Gameplay...
"\Python\Python312\python.exe" train_with_cloning.py
goto end

:record
cls
echo Starting a New Gameplay Recording Session...
echo (Basic version - captures actions only)
echo.
"\Python\Python312\python.exe" imitation_simple.py
goto end

:record_camera
cls
echo Starting ENHANCED Recording with Camera Tracking...
echo (Recommended! Captures actions + camera movements)
echo.
"\Python\Python312\python.exe" imitation_with_camera.py
goto end

:clone
cls
echo Training AI from Scratch (without recorded data)...
echo.
"\Python\Python312\python.exe" main.py
goto end

:analyze
cls
echo Opening AI Learning Analyzer...
echo.
"\Python\Python312\python.exe" analyze.py
goto end

:custom
cls
echo.
set /p script="Enter the Python script to run: "
if "%script%"=="" goto end
"\Python\Python312\python.exe" %script%
goto end

:end
REM Keep window open so you can see output and any errors
echo.
echo Press any key to close this window...
pause >nul

