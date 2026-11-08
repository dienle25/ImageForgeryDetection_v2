@echo off
title Chon nguong DefakerOne2 tren tap hieu chinh - Image Forgery Detection v2
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "APP=%~dp0.."
for %%I in ("%APP%") do set "APP=%%~fI"
cd /d "%APP%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
set "PYTHONHOME="
rem ===================== DUONG DAN - sua cac dong nay neu de du lieu o cho khac =====================
rem Thu muc giai nen data.zip (co that, sua, khac_phuc)
if not defined IFD_DATA_DIR set "IFD_DATA_DIR=C:\Users\ASUS\Downloads\data"
rem Ban goc thu muc khac_phuc (chep vao data\khac_phuc neu chua co)
if not defined IFD_KHAC_PHUC_GOC set "IFD_KHAC_PHUC_GOC=D:\ASUS\2\bo_khac_phuc_data_zip\khac_phuc"
rem Thu muc anh cong khai do tools\tao_nhan_cong_khai.py tao (ngoai repo)
if not defined IFD_CONG_KHAI_DIR set "IFD_CONG_KHAI_DIR=D:\ASUS\2\Image_Forgery_Detection_CPU\du_lieu_cong_khai"
rem Python co pyarrow - chi de dua OpenFake dev vao tap hieu chinh; khong co thi OpenFake bi bo qua
if not defined IFD_PY_PYARROW set "IFD_PY_PYARROW=D:\ASUS\2\Image_Forgery_Detection_CPU\venv_research\Scripts\python.exe"
rem ====================================================================================================
call :find_python
if errorlevel 1 goto :fail
echo Chon nguong DefakerOne2 tren TAP HIEU CHINH rieng (phan dev cua du lieu cong khai, khong dung tap test).
echo Them --chi-anh-that de chi cham anh that, --du-doan FILE de dung du_doan.csv co san, --ghi de ghi nguong vao config.
echo.
"%PY%" -X utf8 "%APP%\tools\chon_nguong_defaker2.py" %*
set "RC=%ERRORLEVEL%"
echo.
echo Ma ket thuc %RC%  -  0: OK, 2: loi dau vao hoac bi chan vi du lieu khong phai tap hieu chinh, 3: server chua san sang.
echo De xuat: reports\defaker2\nguong_de_xuat.json - nhat ky trong reports\defaker2\hieu_chinh_^<ngay gio^>\
if not defined NO_PAUSE pause
exit /b %RC%

:fail
echo.
echo Khong khoi dong duoc. Doc thong bao phia tren.
if not defined NO_PAUSE pause
exit /b 1

:find_python
set "PY="
if exist "%APP%\runtime\python.exe" set "PY=%APP%\runtime\python.exe"
if defined PY exit /b 0
if exist "%APP%\.venv\Scripts\python.exe" set "PY=%APP%\.venv\Scripts\python.exe"
if defined PY exit /b 0
echo [LOI] Khong thay thu muc runtime hoac .venv. Hay chay RUN_APP.bat mot lan truoc - hoac dung ban dong goi day du.
exit /b 1
