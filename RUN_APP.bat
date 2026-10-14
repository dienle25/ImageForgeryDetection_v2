@echo off
title Kiem tra anh - Image Forgery Detection v2
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "APP=%~dp0"
cd /d "%APP%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "PYTHONHOME="
call :find_python
if errorlevel 1 goto :fail
"%PY%" -X utf8 -m ifd.launcher %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0
echo.
echo Ung dung dung voi ma loi %RC%. Doc thong bao phia tren; chup man hinh neu can ho tro.
pause
exit /b %RC%

:fail
echo.
echo Khong khoi dong duoc ung dung. Doc thong bao phia tren.
pause
exit /b 1

:find_python
rem Prefer the bundled runtime, then an existing .venv, else create .venv from an installed Python 3.10-3.13.
set "PY="
if exist "%APP%runtime\python.exe" set "PY=%APP%runtime\python.exe"
if defined PY exit /b 0
if exist "%APP%.venv\Scripts\python.exe" set "PY=%APP%.venv\Scripts\python.exe"
if defined PY exit /b 0
set "SYS_PY="
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined SYS_PY py -%%V -c "import sys" >nul 2>&1 && set "SYS_PY=py -%%V"
)
if not defined SYS_PY python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "SYS_PY=python"
if not defined SYS_PY (
  echo [LOI] Khong co thu muc runtime va khong tim thay Python 3.10 - 3.13 tren may.
  echo       Hay dung ban dong goi day du - co thu muc runtime - hoac cai Python 3.13 tu python.org.
  exit /b 1
)
echo Lan dau chay: tao moi truong .venv bang %SYS_PY% ...
%SYS_PY% -m venv "%APP%.venv"
if errorlevel 1 exit /b 1
set "PY=%APP%.venv\Scripts\python.exe"
if exist "%APP%wheels\runtime\" goto :offline_install
echo Cai thu vien tu PyPI - can Internet mot lan, KHONG gui anh di dau ...
"%PY%" -m pip install --disable-pip-version-check -r "%APP%requirements.txt"
exit /b %ERRORLEVEL%
:offline_install
echo Cai thu vien tu thu muc wheels - khong can Internet ...
"%PY%" -m pip install --disable-pip-version-check --no-index --find-links "%APP%wheels\runtime" -r "%APP%requirements.txt"
exit /b %ERRORLEVEL%
