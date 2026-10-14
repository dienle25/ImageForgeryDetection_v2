@echo off
title Kiem dinh tren anh co nhan cua cong ty - Image Forgery Detection v2
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
"%PY%" -X utf8 -m ifd.company_eval "%~1" --output "%APP%outputs" --open %2 %3 %4 %5
set "RC=%ERRORLEVEL%"
echo.
echo Ma ket thuc %RC%  -  0: OK, 1: co anh khong doc duoc (xem bao cao), 2: loi dau vao/cau hinh.
echo Bao cao nam trong outputs\kiem_dinh - moi lan chay mot thu muc bao_cao_... rieng.
pause
exit /b %RC%

:usage
echo Kiem dinh ung dung tren ANH CO NHAN cua cong ty - anh KHONG roi khoi may.
echo.
echo Chuan bi mot thu muc, ben trong co:
echo     that\   anh that, chua chinh sua  (co the chia thu muc con: that\dien_thoai, that\may_anh ...)
echo     sua\    anh da chinh sua / tao gia (co the chia: sua\ghep_anh, sua\anh_ai, sua\deepfake_mat ...)
echo Roi keo tha THU MUC do vao file KIEM_DINH_CONG_TY.bat nay.
echo Hoac dung file nhan CSV:  KIEM_DINH_CONG_TY.bat "D:\du lieu" --labels "D:\nhan.csv"
echo Xem huong dan chi tiet trong README_VI.md - muc "Kiem dinh tren du lieu cong ty".
pause
exit /b 2

:fail
echo.
echo Khong khoi dong duoc. Doc thong bao phia tren.
pause
exit /b 1

:find_python
set "PY="
if exist "%APP%runtime\python.exe" set "PY=%APP%runtime\python.exe"
if defined PY exit /b 0
if exist "%APP%.venv\Scripts\python.exe" set "PY=%APP%.venv\Scripts\python.exe"
if defined PY exit /b 0
echo [LOI] Khong thay thu muc runtime hoac .venv. Hay chay RUN_APP.bat mot lan truoc - hoac dung ban dong goi day du.
exit /b 1
