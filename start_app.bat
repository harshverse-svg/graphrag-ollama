@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)

if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
)

echo Installing or refreshing dependencies...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Launching GraphRAG Studio...
call ".venv\Scripts\python.exe" -m streamlit run app.py
