#!/bin/bash
# Start Ollama with optimizations and launch the Gradio WebUI

echo "🚀 Starting Ollama with optimizations for Gradio WebUI"

# Check if Python and pip are installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python first."
    exit 1
fi

# Detect if running on macOS
if [[ $(uname) == "Darwin" ]]; then
    IS_MAC=true
    echo "💻 Detected macOS environment - will use Ollama's built-in optimizations"
else
    IS_MAC=false
    echo "🖥️ Detected Linux/Windows environment - will attempt vLLM setup"
fi

# Install required packages
echo "📦 Installing required packages..."
pip install -q gradio==3.50.2 ollama==0.1.6 requests markdown

# Install additional dependencies based on platform
if [[ "$IS_MAC" == true ]]; then
    echo "💻 Installing macOS-compatible dependencies..."
    pip install -q torch transformers
    
    # Set environment variables for macOS optimization
    echo "🔧 Setting up macOS optimization variables"
    export OLLAMA_MODELS="/Users/$(whoami)/.ollama/models"
    export OLLAMA_HOST="127.0.0.1:11434"
else
    # For Linux/Windows, try to install vLLM
    echo "📦 Installing vLLM and its dependencies..."
    pip install -q torch transformers accelerate
    pip install -q vllm
    
    # Check if CUDA is available for PyTorch
    python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" || echo "Warning: CUDA might not be properly configured"
    
    # Set environment variables for vLLM
    echo "🔧 Setting up vLLM environment variables"
    export OLLAMA_USE_VLLM=true
    export OLLAMA_HOST="127.0.0.1:11434"
    export OLLAMA_VLLM_OPTIONS="{\"gpu_memory_utilization\": 0.8, \"max_model_len\": 8192}"
fi

# Check if Ollama is running, start if not
if ! curl -s http://localhost:11434/api/version &> /dev/null; then
    echo "🔄 Starting Ollama service..."
    ollama serve &
    
    # Wait for Ollama to start
    echo "⏳ Waiting for Ollama to start..."
    for i in {1..20}; do
        if curl -s http://localhost:11434/api/version &> /dev/null; then
            break
        fi
        sleep 1
    done
else
    echo "✅ Ollama service is already running"
fi

# Check if qwen3:vpcs-vllm model is available, pull if not
if ! ollama list | grep -q "qwen3:vpcs-vllm"; then
    echo "🔄 Pulling qwen3:latest model first..."
    ollama pull qwen3:latest
    
    # Platform-specific model creation
    if [[ "$IS_MAC" == true ]]; then
        echo "🔧 Creating custom qwen3:vpcs-vllm model with macOS optimizations..."
        
        # Create a Modelfile optimized for macOS
        cat > Modelfile.qwen3-vpcs-vllm << EOL
FROM qwen3:latest

# System prompt optimization
SYSTEM """You are Qwen3, an AI assistant by vpcs-vllm running on macOS. You are helpful, harmless, and honest."""

# Standard Ollama parameters for good performance
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 32768
PARAMETER num_gpu 1
PARAMETER num_thread 8
EOL
    else
        echo "🔧 Creating custom qwen3:vpcs-vllm model with vLLM optimization..."
        
        # Create a Modelfile that is compatible with vLLM
        cat > Modelfile.qwen3-vpcs-vllm << EOL
FROM qwen3:latest

# System prompt optimization
SYSTEM """You are Qwen3, an AI assistant by vpcs-vllm running with vLLM optimization. You are helpful, harmless, and honest."""

# Parameters supported by both vLLM and Ollama
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 32768
EOL
    fi
    
    # Create the model
    ollama create qwen3:vpcs-vllm -f Modelfile.qwen3-vpcs-vllm
    
    echo "✅ Model created successfully with platform-specific optimizations."
else
    echo "✅ qwen3:vpcs-vllm model is already available"
fi

# Platform-specific verification
if [[ "$IS_MAC" == false ]]; then
    # Only verify vLLM on non-Mac platforms
    echo "🔍 Verifying vLLM installation..."
    if python3 -c "import vllm" 2>/dev/null; then
        echo "✅ vLLM is properly installed"
    else
        echo "⚠️ vLLM is not available - will use standard Ollama optimizations"
    fi
fi

# Start the app
echo "🚀 Starting Gradio WebUI with platform-specific optimizations..."
python3 vllm_app.py
