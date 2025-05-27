#!/bin/bash
# Startup script for Ollama Gradio WebUI with Qwen3:vpcs
# Updated version with support for app.py and simple_chat.py

# Set default app to run
APP_TO_RUN="app.py"

# Process command-line arguments
while getopts ":s" opt; do
  case $opt in
    s) # Use simple chat instead of main app
      APP_TO_RUN="simple_chat.py"
      echo "Will run simple_chat.py instead of app.py"
      ;;
    \?) # Invalid option
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# If running in a virtual environment, uncomment and adjust this line
# source /path/to/.venv/bin/activate

# Set base directory to the location of this script
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Function to check command availability
check_command() {
    command -v $1 >/dev/null 2>&1 || { echo "Error: $1 is required but not installed. Aborting."; exit 1; }
}

# Check for required commands
check_command ollama
check_command pip
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" || { echo "Error: Python 3.8 or higher is required."; exit 1; }

# Ensure Ollama server is running
ps aux | grep ollama | grep -v grep > /dev/null
if [ $? -ne 0 ]; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3  # Give it a bit more time to start
    echo "Checking if Ollama server started successfully..."
    ps aux | grep ollama | grep -v grep > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Failed to start Ollama server. Please check your installation."
        exit 1
    fi
fi

# Check if Qwen3:vpcs model exists
echo "Checking for Qwen3:vpcs model..."
ollama list | grep qwen3:vpcs > /dev/null
if [ $? -ne 0 ]; then
    echo "Creating Qwen3:vpcs model with extended context..."
    mkdir -p qwen3-custom
    echo "from qwen3:latest" > qwen3-custom/Modelfile
    echo "parameter num_ctx 32768" >> qwen3-custom/Modelfile
    ollama create -f qwen3-custom/Modelfile qwen3:vpcs
    
    # Verify model was created successfully
    ollama list | grep qwen3:vpcs > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create Qwen3:vpcs model. Check if qwen3:latest is available."
        echo "Available models:"
        ollama list
        exit 1
    fi
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if prompt.json exists, create it if not
if [ ! -f "prompt.json" ]; then
    echo "Creating default prompt.json..."
    echo '{"React Agent": "You are an expert Odoo developer assistant that follows the REACT pattern:\n1. Reason about what needs to be done\n2. Choose an Action to take\n3. Observe the result and continue reasoning\n\nYour goal is to help users generate high-quality Odoo modules based on their requirements.\nWhen responding to the user, be concise, clear and professional."}' > prompt.json
fi

# Start the Gradio app
echo "Starting Ollama Gradio WebUI with $APP_TO_RUN..."
python "$APP_TO_RUN"

# Provide message if app crashes
if [ $? -ne 0 ]; then
    echo "Error: The application crashed. Check the error messages above."
    echo "You can try running the simple chat version with: ./start.sh -s"
    exit 1
fi
