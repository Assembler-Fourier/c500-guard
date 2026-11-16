@echo off
REM Forced realignment helper for the C500 Guard.
REM
REM Double-click this when the guard says "Scene alignment lost". It brings the
REM camera up on its own - without the guard, so nothing can time out and shut
REM it down again - and shows a live score while you move the camera. It locks
REM the alignment by itself the moment the view fits, keeping all alarm zones.
REM
REM The video path runs elevated, so this asks for administrator rights.

set PY=C:\c500-guard\.venv\Scripts\python.exe
set TOOL=C:\c500-guard\src\realign.py
set MODE=%1
if "%MODE%"=="" set MODE=--aim

net session >nul 2>&1
if %errorlevel%==0 goto run
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -FilePath '%~f0' -ArgumentList '%MODE%' -Verb RunAs"
exit /b

:run
title C500 - forced realignment
"%PY%" "%TOOL%" %MODE%
echo.
echo ---------------------------------------------------------------
echo Finished. This window stays open so you can read what happened.
echo ---------------------------------------------------------------
pause
