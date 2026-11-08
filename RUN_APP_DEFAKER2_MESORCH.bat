@echo off
rem DefakerOne2 lam MO HINH CHINH + Mesorch hien ban do vung nghi sua (tham khao, khong gop diem).
rem Community Forensics va face_clip tat de tiet kiem RAM. Can bat addons\defaker2\CHAY_DEFAKER2.bat truoc.
call "%~dp0RUN_APP.bat" --config research\configs\config_defaker2_mesorch.json %*
exit /b %ERRORLEVEL%
