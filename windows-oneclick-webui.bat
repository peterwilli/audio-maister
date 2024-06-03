@echo off
SET venv_dir=venv
SET pyfile=main.py
SET python=%venv_dir%\Scripts\python.exe

REM Check if the virtual environment directory exists
IF EXIST "%venv_dir%\Scripts\activate.bat" (
    ECHO Virtual environment found. Activating...
    CALL %venv_dir%\Scripts\activate.bat
) ELSE (
    ECHO Creating virtual environment...
    python -m venv %venv_dir%
    CALL %venv_dir%\Scripts\activate.bat
)

REM Check if the virtual environment is activated
IF NOT "%VIRTUAL_ENV%" == "" (
    ECHO Virtual environment activated.
    ECHO Installing dependencies...
    %python% -m pip install --upgrade pip
    %python% -m pip install -r requirements.txt 
    %python% -m pip install gradio==3.50.2
    ECHO Dependencies installed.
    
    REM Run the Python script
    python app.py
) ELSE (
    ECHO Failed to activate virtual environment.
)

REM Pause the command window
pause
