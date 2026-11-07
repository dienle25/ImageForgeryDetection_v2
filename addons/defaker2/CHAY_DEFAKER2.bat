@echo off
title DefakerOne2 server (thu nghiem) - 127.0.0.1:8765
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

rem ===================== DUONG DAN - sua cac dong nay neu cai o cho khac =====================
rem Ma nguon DefakerOne2 (thu muc co logprob2score.py)
set "DEFAKER2_REPO=D:\deep_fake\DefakerOne2"
rem Trong so release/ (thu muc co config.json va model.safetensors)
set "DEFAKER2_MODEL=D:\deep_fake\models\DefakerOne2\release"
rem Anaconda/Miniconda va ten moi truong conda rieng
set "CONDA_ROOT=D:\2_Softwares\anaconda3"
set "CONDA_ENV=defaker2"
rem Cong server (ung dung goi http://127.0.0.1:8765/score)
set "DEFAKER2_PORT=8765"
rem ==========================================================================================

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "MIN_PIXELS=262144"
set "MAX_PIXELS=2408448"
set "HF_HUB_OFFLINE=1"

if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
  echo [LOI] Khong thay conda tai "%CONDA_ROOT%". Sua dong CONDA_ROOT o dau file nay.
  goto :fail
)
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV%"
if errorlevel 1 (
  echo [LOI] Khong kich hoat duoc moi truong conda "%CONDA_ENV%".
  goto :fail
)
echo Da kich hoat moi truong conda: %CONDA_ENV%
echo Ma nguon : %DEFAKER2_REPO%
echo Trong so : %DEFAKER2_MODEL%
echo Dia chi server: http://127.0.0.1:%DEFAKER2_PORT%/score
echo Dang nap mo hinh (khoang 1-2 phut). Cho dong "DA NAP XONG" roi moi phan tich anh.
echo Them --device cpu de chay CPU (bf16, can khoang 4.5 GB RAM trong, rat cham).
echo.
set "TRY=1"
:run_server
python -X utf8 "%~dp0defaker2_server.py" --repo "%DEFAKER2_REPO%" --model "%DEFAKER2_MODEL%" --port %DEFAKER2_PORT% %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0
rem -1073741819 = 0xC0000005: on Windows usually not enough virtual memory (RAM + pagefile) while loading
if not "%RC%"=="-1073741819" goto :stopped
if not "%TRY%"=="1" goto :stopped
echo.
echo [CANH BAO] Loi bo nho 0xC0000005 khi nap mo hinh. Tren Windows thuong do thieu bo nho ao - RAM + pagefile:
echo            server can khoang 7-8 GB bo nho ao du chi dung khoang 1-2 GB RAM that.
echo            Windows thuong tu noi pagefile sau lan loi dau. Thu lai sau 10 giay ...
set "TRY=2"
ping -n 11 127.0.0.1 >nul
goto :run_server

:stopped
echo.
echo Server dung voi ma loi %RC%. Doc thong bao phia tren.
if "%RC%"=="-1073741819" echo Van loi 0xC0000005: dong bot ung dung nang, de trong it nhat 10 GB tren o C: cho pagefile roi chay lai.
pause
exit /b %RC%

:fail
pause
exit /b 1
