@echo off
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%\lab2_env"
call "%VENV_DIR%\Scripts\activate"
streamlit run "%PROJECT_DIR%\main.py"
pause
call deactivate