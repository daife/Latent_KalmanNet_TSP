@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0reproduce_paper_gpu.ps1" %*
exit /b %ERRORLEVEL%
