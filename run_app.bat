@echo off
setlocal

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Virtual environment belum ada di .venv\Scripts\python.exe
  echo Jalankan dulu pembuatan venv dan install dependency.
  exit /b 1
)

"%PYTHON_EXE%" -m streamlit run "%~dp0app.py"
