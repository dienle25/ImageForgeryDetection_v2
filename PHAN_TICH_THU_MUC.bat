@echo off
title Phan tich ca thu muc - Image Forgery Detection v2
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
if "%~1"=="" goto :usage
"%PY%" -X utf8 -m ifd "%~1" --output "%APP%outputs" %2 %3 %4 %5
set "RC=%ERRORLEVEL%"
echo.
echo Ma ket thuc %RC%  -  0: tat ca OK, 1: co anh loi, 2: loi dau vao/cau hinh.
echo Ket qua nam trong thu muc outputs\runs - moi lan chay mot thu muc rieng.
pause
exit /b %RC%

:usage
echo Keo tha mot THU MUC anh - hoac mot file anh - vao file PHAN_TICH_THU_MUC.bat nay.
echo Hoac chay:  PHAN_TICH_THU_MUC.bat "D:\duong dan\thu muc anh"
pause
exit /b 2

:fail
echo.
echo Khong khoi dong duoc. Doc thong bao phia tren.
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
