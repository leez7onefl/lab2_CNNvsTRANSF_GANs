@echo off
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%\lab2_env"
set "REQUIREMENTS_FILE=%PROJECT_DIR%\requirements.txt"

IF EXIST "%VENV_DIR%" (
    echo erasing existing venv
    rmdir /s /q "%VENV_DIR%"
)

echo creating venv...
python -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate"
python -m pip install -r "%REQUIREMENTS_FILE%"
call deactivate

echo venv created. Press enter to exit.
pause