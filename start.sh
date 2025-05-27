#!/bin/bash
# Startup script for Ollama Gradio WebUI with Qwen3:vpcs

# If running in a virtual environment, uncomment and adjust this line
# source /path/to/.venv/bin/activate

# Ensure Ollama server is running
ps aux | grep ollama | grep -v grep > /dev/null
if [ $? -ne 0 ]; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 2
fi

# Check if Qwen3:vpcs model exists
ollama list | grep qwen3:vpcs > /dev/null
if [ $? -ne 0 ]; then
    echo "Creating Qwen3:vpcs model with extended context..."
    mkdir -p qwen3-custom
    echo "from qwen3:latest" > qwen3-custom/Modelfile
    echo "parameter num_ctx 32768" >> qwen3-custom/Modelfile
    ollama create -f qwen3-custom/Modelfile qwen3:vpcs
fi

# Install dependencies if needed
pip install -r requirements.txt

# Start the Gradio app
echo "Starting Ollama Gradio WebUI..."
python app.py
