@echo off
rem Builds the offline bundle for an industrial PC without internet:
rem   release\SegSynth-<version>-offline-win_amd64.zip
rem Run it on a PC that can download Python packages. Copy the zip to the target PC,
rem unzip it there and double-click start.bat.
setlocal
cd /d "%~dp0"
set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 set "PY=py -3"
)
if not defined PY set "PY=python"
%PY% -m pip install --disable-pip-version-check -q wheel setuptools
%PY% packaging\build_offline_bundle.py --platform win_amd64 %*
if errorlevel 1 (
  echo.
  echo  Building the bundle failed. See the messages above.
  pause
  exit /b 1
)
echo.
echo  Done. The bundle is in the release folder.
explorer release
pause
