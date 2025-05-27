@echo off
setlocal enabledelayedexpansion

:: Set default app to run
set APP_TO_RUN=app.py

:: Process command-line arguments
if "%1"=="-s" (
    set APP_TO_RUN=simple_chat.py
    echo Will run simple_chat.py instead of app.py
)

:: Set base directory to the location of this script
cd /d "%~dp0"

:: Check if conda environment exists and activate it
call conda activate ollama 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda environment 'ollama' not found.
    echo Please create it with: conda create -n ollama python=3.8
    pause
    exit /b 1
)

:: Check if Ollama is installed
where ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Ollama is not installed or not in PATH.
    echo Please install Ollama from: https://github.com/ollama/ollama
    pause
    exit /b 1
)

:: Check if Ollama server is running
tasklist /fi "imagename eq ollama.exe" | find "ollama.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Starting Ollama server...
    start /b ollama serve
    timeout /t 3 >nul
    
    :: Verify server started
    tasklist /fi "imagename eq ollama.exe" | find "ollama.exe" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to start Ollama server. Please check your installation.
        pause
        exit /b 1
    )
)

:: Check if Qwen3:vpcs model exists
echo Checking for Qwen3:vpcs model...
ollama list | find "qwen3:vpcs" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Creating Qwen3:vpcs model with extended context...
    if not exist qwen3-custom mkdir qwen3-custom
    echo from qwen3:latest > qwen3-custom\Modelfile
    echo parameter num_ctx 32768 >> qwen3-custom\Modelfile
    ollama create -f qwen3-custom\Modelfile qwen3:vpcs
    
    :: Verify model was created successfully
    ollama list | find "qwen3:vpcs" >nul
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create Qwen3:vpcs model. Check if qwen3:latest is available.
        echo Available models:
        ollama list
        pause
        exit /b 1
    )
)

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Check if prompt.json exists, create it if not
if not exist prompt.json (
    echo Creating default prompt.json...
    echo {"React Agent": "You are an expert Odoo developer assistant that follows the REACT pattern:\n1. Reason about what needs to be done\n2. Choose an Action to take\n3. Observe the result and continue reasoning\n\nYour goal is to help users generate high-quality Odoo modules based on their requirements.\nWhen responding to the user, be concise, clear and professional."} > prompt.json
)

:: Start the Gradio app
echo Starting Ollama Gradio WebUI with %APP_TO_RUN%...
python "%APP_TO_RUN%"

:: Check if app crashed
if %ERRORLEVEL% NEQ 0 (
    echo Error: The application crashed. Check the error messages above.
    echo You can try running the simple chat version with: start.bat -s
    pause
    exit /b 1
)

pause