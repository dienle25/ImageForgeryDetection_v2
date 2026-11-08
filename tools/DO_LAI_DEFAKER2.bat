@echo off
title Do lai DefakerOne2 - Image Forgery Detection v2
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
if "%~1"=="" goto :usage
"%PY%" -X utf8 "%APP%\tools\do_lai_defaker2.py" %*
set "RC=%ERRORLEVEL%"
echo.
echo Ma ket thuc %RC%  -  0: OK, 1: co anh loi (xem tong_hop.md), 2: loi dau vao/cau hinh, 3: server DefakerOne2 chua san sang.
echo Ket qua va nhat ky: reports\defaker2\ - gui file nhat_ky.log neu can ho tro.
if not defined NO_PAUSE pause
exit /b %RC%

:usage
echo Do lai DefakerOne2 tren du lieu doc lap. Can server addons\defaker2\CHAY_DEFAKER2.bat dang chay.
echo.
echo   tools\DO_LAI_DEFAKER2.bat A                 bo A (hoac B, C, D, cocoglide, korus, tat_ca)
echo   tools\DO_LAI_DEFAKER2.bat A --thu 10        chay thu 10 anh dau
echo   tools\DO_LAI_DEFAKER2.bat korus --mau 100   mau ngau nhien theo nhom (seed mac dinh 20261108, doi bang --seed)
echo   them --chi-defaker2 de chi chay DefakerOne2, khong bat Mesorch
echo Ket qua: reports\defaker2\^<bo^>_^<ngay gio^>\tong_hop.md
if not defined NO_PAUSE pause
exit /b 2

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
