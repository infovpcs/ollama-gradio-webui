#!/bin/sh
# Script to install dependencies for Ollama Gradio WebUI with vLLM
# Compatible with local environments and Kaggle notebooks

echo "🚀 Installing dependencies for Ollama Gradio WebUI with vLLM"

# Detect environment
IS_KAGGLE=false
if [ -d "/kaggle" ]; then
    IS_KAGGLE=true
    echo "📊 Detected Kaggle notebook environment"
elif [ "$(uname)" = "Darwin" ]; then
    IS_MAC=true
    echo "📱 Detected macOS environment"
else
    IS_LINUX=true
    echo "🖥️ Detected Linux/Windows environment"
fi

# Skip virtual environment creation on Kaggle
if [ "$IS_KAGGLE" = false ] && [ -z "${VIRTUAL_ENV}" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
    . .venv/bin/activate
fi

# Install basic dependencies first
echo "📦 Installing basic dependencies..."
pip install -q gradio==3.50.2 ollama==0.1.6 pydantic==1.10.8 requests markdown numpy

# Install PyTorch based on environment
if [ "$IS_KAGGLE" = true ]; then
    echo "✅ Using pre-installed PyTorch in Kaggle environment"
else
    echo "🔥 Installing PyTorch (required before vLLM)..."
    if [ "$IS_MAC" = true ]; then
        pip install -q torch torchvision torchaudio
    else
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
fi

# Install Hugging Face libraries
echo "🤗 Installing Hugging Face libraries..."
pip install -q transformers>=4.33.0 accelerate>=0.23.0

# Install vLLM and related dependencies based on environment
if [ "$IS_KAGGLE" = true ]; then
    echo "📦 Installing Kaggle-specific dependencies..."
    
    # Install additional dependencies that might be needed
    pip install -q einops>=0.6.1 sentencepiece bitsandbytes
    
    # Install xformers for better performance if not already installed
    if ! python -c "import xformers" 2>/dev/null; then
        echo "📦 Installing xformers in Kaggle environment..."
        pip install -q xformers
    fi
    
    # Check if vLLM is already installed in Kaggle
    if python -c "import vllm" 2>/dev/null; then
        echo "✅ vLLM is already installed in Kaggle environment"
    else
        echo "📦 Installing vLLM in Kaggle environment..."
        pip install -q vllm
    fi
    
    echo "✅ Kaggle setup complete! Both notebook and server modes are supported."
elif [ "$IS_MAC" = true ]; then
    echo "📦 Installing macOS compatible dependencies..."
    
    # Mac users should use Ollama's built-in optimizations instead of vLLM
    # But we can install minimal components for compatibility
    pip install -q einops>=0.6.1 sentencepiece ray
    
    echo "⚠️ Note: vLLM with GPU acceleration is not fully supported on macOS."
    echo "⚠️ Using Ollama's built-in optimizations instead."
else
    echo "📦 Installing vLLM and related dependencies for Linux/Windows..."
    
    # Install prerequisites
    pip install -q einops>=0.6.1 sentencepiece ray
    
    # First install xformers separately to avoid build issues
    pip install -q xformers bitsandbytes
    
    # Then install vLLM
    pip install -q vllm>=0.2.0
    
    echo "✅ vLLM installed with GPU support"
fi

echo "✅ All dependencies installed successfully!"
echo "📝 Note: For Kaggle integration, ensure 'Internet' is enabled in the notebook settings."
echo "🔧 Run ./start_vllm.sh to start the application with vLLM optimizations."

