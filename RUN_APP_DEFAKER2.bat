@echo off
rem DefakerOne2 lam MO HINH CHINH (3 bo cu tat). Can bat addons\defaker2\CHAY_DEFAKER2.bat truoc.
call "%~dp0RUN_APP.bat" --config research\configs\config_defaker2_chinh.json %*
exit /b %ERRORLEVEL%
