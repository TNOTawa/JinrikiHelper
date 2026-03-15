@echo off
chcp 65001 >nul
echo 启动人力V助手 (便携版)...
set PYTHONPATH=%~dp0

set MFA_ROOT_DIR=%~dp0mfa_data
set PATH=%PATH%;%~dp0tools\ffmpeg\bin

"%~dp0python\python.exe" "%~dp0main.py"
pause