@echo off
rem SegSynth - double-click to start from the source folder.
rem The first start creates .venv and installs SegSynth and its dependencies with pip
rem (needs internet or your company package mirror). "start.bat debug" keeps a console open.
setlocal
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" (
  "%VPY%" -c "import segsynth, cv2, PySide6, numpy" >nul 2>nul
  if not errorlevel 1 goto :launch
)
call :install
if errorlevel 1 goto :fail

:launch
if /i "%~1"=="debug" (
  "%VPY%" -m segsynth
  pause
  exit /b
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m segsynth
exit /b 0

:install
echo.
echo  SegSynth - first start, installing...
echo.
set "PY="
where py >nul 2>nul
if not errorlevel 1 (
  for %%v in (3.12 3.13 3.11 3.10) do if not defined PY call :try_python "py -%%v"
)
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 call :try_python "python"
)
if not defined PY (
  echo  Python 3.10 - 3.13 ^(64-bit^) was not found. Install it and start this file again.
  exit /b 1
)
if not exist "%VPY%" (
  %PY% -m venv .venv
  if errorlevel 1 exit /b 1
)
"%VPY%" -m pip install --disable-pip-version-check -e .
if errorlevel 1 exit /b 1
echo  Installed.
exit /b 0

:fail
echo.
echo  SegSynth could not be installed. See the messages above.
pause
exit /b 1

:try_python
%~1 -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=%~1"
exit /b 0
