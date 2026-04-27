@echo off
setlocal
cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
  echo Creating virtual environment...
  cd backend
  python -m venv .venv
  cd ..
)

if not exist "backend\.env" (
  copy /Y "backend\.env.example" "backend\.env" >nul
)

echo Installing or refreshing dependencies...
call "backend\.venv\Scripts\python.exe" -m pip install -r backend\requirements.txt

echo Launching GraphRAG Studio...
call "backend\.venv\Scripts\python.exe" -m streamlit run frontend\app.py
