#!/bin/sh
# Start Ollama with optimizations and launch the Gradio WebUI
# Compatible with local environments and Kaggle notebooks

echo "🚀 Starting Ollama with optimizations for Gradio WebUI"

# Ensure we use the right Python command on Kaggle
PYTHON_CMD="python"
if command -v python3 > /dev/null 2>&1; then
    PYTHON_CMD="python3"
fi

# Check if Python is working properly
if ! $PYTHON_CMD -c "print('Python is working')" > /dev/null 2>&1; then
    # Try with just python if python3 fails
    PYTHON_CMD="python"
    if ! $PYTHON_CMD -c "print('Python is working')" > /dev/null 2>&1; then
        echo "❌ Cannot execute Python. Please check your environment."
        exit 1
    fi
fi

# Detect environment
IS_KAGGLE=false
if [ -d "/kaggle" ]; then
    IS_KAGGLE=true
    echo "📊 Detected Kaggle notebook environment - optimizing for Kaggle"
elif [ "$(uname)" = "Darwin" ]; then
    IS_MAC=true
    echo "💻 Detected macOS environment - will use Ollama's built-in optimizations"
else
    IS_LINUX=true
    echo "🖥️ Detected Linux environment - will attempt vLLM setup"
fi

# Special handling for Kaggle environment
if [ "$IS_KAGGLE" = true ]; then
    echo "📦 Setting up Kaggle-specific configuration..."
    
    # Check if required packages are already installed (most should be in Kaggle)
    if ! $PYTHON_CMD -c "import gradio" 2>/dev/null; then
        pip install -q gradio==3.50.2
    fi
    
    if ! $PYTHON_CMD -c "import ollama" 2>/dev/null; then
        pip install -q ollama==0.1.6
    fi
    
    # Additional requirements
    pip install -q requests markdown numpy
    
    # Set environment variables for Kaggle with vLLM
    echo "🔧 Setting up Kaggle optimization variables"
    export OLLAMA_USE_VLLM=true
    export OLLAMA_HOST="127.0.0.1:11434"
    export OLLAMA_VLLM_OPTIONS="{\"gpu_memory_utilization\": 0.9, \"max_model_len\": 16384}"
    
    # Verify CUDA availability in Kaggle
    $PYTHON_CMD -c "import torch; print(f'CUDA Available in Kaggle: {torch.cuda.is_available()}')" || echo "Warning: CUDA not available in this Kaggle session"
elif [ "$IS_MAC" = true ]; then
    echo "💻 Installing macOS-compatible dependencies..."
    pip install -q gradio==3.50.2 ollama==0.1.6 requests markdown
    pip install -q torch transformers
    
    # Set environment variables for macOS optimization
    echo "🔧 Setting up macOS optimization variables"
    export OLLAMA_MODELS="/Users/$(whoami)/.ollama/models"
    export OLLAMA_HOST="127.0.0.1:11434"
else
    # For Linux, install vLLM and dependencies
    echo "📦 Installing Linux dependencies..."
    pip install -q gradio==3.50.2 ollama==0.1.6 requests markdown
    pip install -q torch transformers accelerate
    pip install -q vllm
    
    # Check if CUDA is available for PyTorch
    $PYTHON_CMD -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" || echo "Warning: CUDA might not be properly configured"
    
    # Set environment variables for vLLM
    echo "🔧 Setting up vLLM environment variables"
    export OLLAMA_USE_VLLM=true
    export OLLAMA_HOST="127.0.0.1:11434"
    export OLLAMA_VLLM_OPTIONS="{\"gpu_memory_utilization\": 0.8, \"max_model_len\": 8192}"
fi

# Check Ollama server availability based on environment
if [ "$IS_KAGGLE" = true ]; then
    echo "📊 Configuring for Kaggle notebook environment..."
    
    # In Kaggle, we might need to use a different approach for Ollama
    # Since Ollama server might not be directly installable in all Kaggle environments
    if ! command -v ollama > /dev/null 2>&1; then
        echo "⚠️ Ollama binary not found in Kaggle environment"
        echo "🛠️ Setting up alternative mode with API emulation..."
        
        # Flag to indicate we're in Kaggle API emulation mode
        export KAGGLE_API_MODE=true
        
        # Run vllm_app.py without requiring Ollama server
        echo "🚀 Will run in Kaggle-optimized mode (API-only)"
    else
        # If Ollama is available in Kaggle, try to start it
        echo "🔄 Attempting to start Ollama service in Kaggle..."
        ollama serve &
        sleep 5
        
        if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo "⚠️ Could not start Ollama in Kaggle environment"
            echo "🛠️ Falling back to API-only mode"
            export KAGGLE_API_MODE=true
        else
            echo "✅ Ollama service started successfully in Kaggle"
        fi
    fi
else
    # Standard environment (non-Kaggle) Ollama server check
    if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "🔄 Starting Ollama service..."
        ollama serve &
        
        # Wait for Ollama to start
        echo "⏳ Waiting for Ollama to start..."
        i=1
        max_attempts=20
        while [ $i -le $max_attempts ]; do
            if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
                echo "✅ Ollama service started successfully"
                break
            fi
            echo "Attempt $i: Waiting for Ollama to start..."
            sleep 2
            
            if [ $i -eq $max_attempts ]; then
                echo "❌ Ollama service failed to start in time. Please check your installation."
                exit 1
            fi
            i=$((i+1))
        done
    else
        echo "✅ Ollama service is already running"
    fi
fi

# Model creation/verification based on environment
if [ "$IS_KAGGLE" = true ] && [ "$KAGGLE_API_MODE" = true ]; then
    echo "📊 Using API-only mode in Kaggle (no model creation needed)"
    # Skip model creation in API-only mode
else
    # Check if qwen3:vpcs-vllm model is available, pull if not
    if ! ollama list | grep -q "qwen3:vpcs-vllm"; then
        echo "🔄 Pulling qwen3:latest model first..."
        ollama pull qwen3:latest
        
        # Environment-specific model creation
        if [ "$IS_KAGGLE" = true ]; then
            echo "🔧 Creating custom qwen3:vpcs-vllm model with Kaggle optimizations..."
            
            # Create a Modelfile optimized for Kaggle
            cat > Modelfile.qwen3-vpcs-vllm << EOL
FROM qwen3:latest

# System prompt optimization
SYSTEM """You are Qwen3, an AI assistant by VPCS running in Kaggle with vLLM optimization. You are helpful, harmless, and honest."""

# Parameters for optimal Kaggle performance
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 16384
EOL
        elif [ "$IS_MAC" = true ]; then
            echo "🔧 Creating custom qwen3:vpcs-vllm model with macOS optimizations..."
            
            # Create a Modelfile optimized for macOS
            cat > Modelfile.qwen3-vpcs-vllm << EOL
FROM qwen3:latest

# System prompt optimization
SYSTEM """You are Qwen3, an AI assistant by VPCS running on macOS. You are helpful, harmless, and honest."""

# Standard Ollama parameters for good performance
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 16384
PARAMETER num_gpu 1
PARAMETER num_thread 8
EOL
        else
            echo "🔧 Creating custom qwen3:vpcs-vllm model with vLLM optimization..."
            
            # Create a Modelfile that is compatible with vLLM
            cat > Modelfile.qwen3-vpcs-vllm << EOL
FROM qwen3:latest

# System prompt optimization
SYSTEM """You are Qwen3, an AI assistant by VPCS running with vLLM optimization. You are helpful, harmless, and honest."""

# Parameters supported by both vLLM and Ollama
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 16384
EOL
        fi
        
        # Create the model
        ollama create qwen3:vpcs-vllm -f Modelfile.qwen3-vpcs-vllm
        
        echo "✅ Model created successfully with environment-specific optimizations."
    else
        echo "✅ qwen3:vpcs-vllm model is already available"
    fi
fi

# Environment-specific verification and startup
if [ "$IS_KAGGLE" = true ]; then
    # Special handling for Kaggle environment
    echo "🔍 Verifying Kaggle-specific dependencies..."
    
    # Verify vLLM installation in Kaggle
    if $PYTHON_CMD -c "import vllm" 2>/dev/null; then
        echo "✅ vLLM is properly installed in Kaggle"
    else
        echo "⚠️ vLLM is not available in Kaggle - will use standard mode"
    fi
    
    # Set Kaggle-specific environment variables
    export GRADIO_SERVER_NAME="0.0.0.0"
    export GRADIO_SERVER_PORT="7860"
    
    echo "🚀 Starting Gradio WebUI in Kaggle environment..."
    if [ "$KAGGLE_API_MODE" = true ]; then
        echo "⚠️ Running in API emulation mode (no Ollama server)"
        # Pass Kaggle API mode flag to the app
        $PYTHON_CMD vllm_app.py --kaggle-mode
    else
        $PYTHON_CMD vllm_app.py
    fi
elif [ "$IS_MAC" = true ]; then
    echo "🔍 Verifying macOS dependencies..."
    echo "🚀 Starting Gradio WebUI with macOS optimizations..."
    $PYTHON_CMD vllm_app.py
else
    # Linux environment
    echo "🔍 Verifying vLLM installation..."
    if $PYTHON_CMD -c "import vllm" 2>/dev/null; then
        echo "✅ vLLM is properly installed"
    else
        echo "⚠️ vLLM is not available - will use standard Ollama optimizations"
    fi
    
    echo "🚀 Starting Gradio WebUI with Linux optimizations..."
    $PYTHON_CMD vllm_app.py
fi
