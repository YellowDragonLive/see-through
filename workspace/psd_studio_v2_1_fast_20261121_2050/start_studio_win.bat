@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not defined PY (
  for %%P in (
    "%USERPROFILE%\miniforge3\envs\seethrough\python.exe"
    "%USERPROFILE%\miniforge3\envs\see_through\python.exe"
    "%USERPROFILE%\miniconda3\envs\seethrough\python.exe"
    "%USERPROFILE%\miniconda3\envs\see_through\python.exe"
    "%USERPROFILE%\anaconda3\envs\seethrough\python.exe"
    "%USERPROFILE%\anaconda3\envs\see_through\python.exe"
  ) do (
    if not defined PY if exist %%~P set "PY=%%~P"
  )
)
if not defined PY (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PY set "PY=%%P"
  )
)
if not defined PY (
  echo [FAIL] Python not found. Set PY=C:\path\to\python.exe and retry.
  exit /b 1
)

"%PY%" "%~dp0psd_studio.py" %*
exit /b %ERRORLEVEL%
