#!/bin/bash
# Script to install dependencies for Ollama Gradio WebUI with vLLM

echo "🚀 Installing dependencies for Ollama Gradio WebUI with vLLM"

# Detect if running on macOS
if [[ $(uname) == "Darwin" ]]; then
    IS_MAC=true
    echo "📱 Detected macOS environment"
else
    IS_MAC=false
    echo "🖥️ Detected Linux/Windows environment"
fi

# Create and activate virtual environment if not already in one
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Install basic dependencies first
echo "📦 Installing basic dependencies..."
pip install -q gradio==3.50.2 ollama==0.1.6 pydantic==1.10.8 requests markdown numpy

# Install PyTorch first (crucial for xformers)
echo "🔥 Installing PyTorch (required before vLLM)..."
if [[ "$IS_MAC" == true ]]; then
    pip install torch torchvision torchaudio
else
    # For Linux/Windows with CUDA, use the appropriate PyTorch version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install transformers and accelerate
echo "🤗 Installing Hugging Face libraries..."
pip install -q transformers>=4.33.0 accelerate>=0.23.0

# Install vLLM if not on Mac, or with special handling for Mac
if [[ "$IS_MAC" == true ]]; then
    echo "⚠️ vLLM with GPU acceleration is not fully supported on macOS."
    echo "⚠️ Installing CPU-compatible alternatives..."
    
    # Mac users should use Ollama's built-in optimizations instead of vLLM
    # But we can install minimal components for compatibility
    pip install -q einops>=0.6.1 sentencepiece ray
else
    echo "🚀 Installing vLLM with GPU support..."
    # First install xformers separately to avoid build issues
    pip install -q xformers
    # Then install vLLM
    pip install -q vllm>=0.2.0
fi

echo "✅ All dependencies installed!"
echo "📝 Note: On macOS, vLLM with GPU acceleration is not fully supported."
echo "🔧 Run ./start_vllm.sh to start the application."
