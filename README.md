# Ollama Gradio WebUI with vLLM

This project provides a user-friendly chat interface using Gradio for interacting with various Ollama models, with special support for Qwen3 models using vLLM acceleration and React agent integration specifically designed for Odoo development. The vLLM backend provides 2-4x faster inference and extended context window capabilities.

## Features

* Chat with multiple Ollama models with vLLM acceleration (2-4x faster inference)
* Extended context window support (8192 tokens) for Qwen3:vpcs model
* Performance tracking and benchmarking for model responses
* React Agent API optimized for Odoo development assistance
* Visual Assistant for image analysis with LLaVA model
* Hindi translation capability for multilingual support
* Multiple prompt templates for different use cases
* Web-based interface with public sharing capability
* File upload support for context-rich interactions

## Applications

This repository contains three main applications:

1. **app.py** - The full-featured application with all capabilities (chat, React agent, translation, visual processing)
2. **vllm_app.py** - Enhanced version with vLLM optimization for faster inference and performance tracking
3. **simple_chat.py** - A simplified version focused only on chat functionality

## How to Run

### Using the Startup Scripts (Recommended)

The easiest way to run the application is using the provided startup scripts:

```bash
# Run the standard application
./start.sh

# Run the simplified chat version
./start.sh -s

# Run the vLLM-optimized version for faster inference
./start_vllm.sh
```

The standard startup script automatically:
- Checks if Ollama server is running and starts it if needed
- Ensures the Qwen3:vpcs model with extended context window is created
- Installs dependencies
- Creates a default prompt.json file if needed

The vLLM startup script (`start_vllm.sh`) additionally:
- Sets the necessary environment variables for vLLM optimization (OLLAMA_USE_VLLM=true)
- Configures optimal GPU memory utilization (OLLAMA_VLLM_GPU_MEMORY_UTILIZATION=0.8)
- Sets up extended context window support (OLLAMA_VLLM_MAX_MODEL_LEN=8192)
- Launches the optimized vLLM interface with performance tracking

### Manual Setup

1. Install [Ollama](https://github.com/ollama/ollama)
2. Create conda environment: `conda create -n ollama python=3.8`
3. Activate conda environment: `conda activate ollama`
4. Install required packages: `pip install -r requirements.txt`
5. Run the application: `python app.py` or `python simple_chat.py`

## Dependencies

This project has specific version requirements to ensure compatibility:

```
gradio==3.50.2
ollama==0.1.6
pydantic==1.10.8
fastapi<0.100.0
starlette<0.28.0
typing-extensions<4.7.0
requests>=2.25.0
markdown>=3.3.0
numpy>=1.19.0
# Optional dependencies for vLLM integration
# Note: vLLM itself is not required on the client side
# as it's used by Ollama server
```

## React Agent Integration

The Ollama Gradio WebUI includes a dedicated React Agent API for Odoo development. This agent follows the Reason-Act-Observe pattern to help users generate high-quality Odoo modules based on their requirements.

API endpoints are available at:
* `/chat` - For direct chat interactions with any model
* `/react_agent` - For Odoo development assistance with the React agent

## Translation Feature

The application includes an intelligent translation feature that can:
- Translate English (or any non-Hindi language) to Hindi
- Translate Hindi to English

This is available in the Visual Assistant tab by enabling the "Enable Hindi Translation" checkbox.

## vLLM Integration

The `vllm_app.py` application features enhanced integration with vLLM-accelerated Ollama models:

### Performance Benefits

- **Faster Inference**: 2-4x faster response generation compared to standard Ollama
- **Extended Context**: Support for up to 8192 token context windows (vs standard 4096)
- **Better Memory Utilization**: Optimized GPU memory usage with configurable parameters
- **Performance Tracking**: Real-time statistics on generation speed (tokens/second)

### How vLLM Works

vLLM (Very Large Language Model) is a high-performance inference engine that:

1. Uses PagedAttention for efficient memory management
2. Implements continuous batching for faster throughput
3. Supports model quantization for reduced memory footprint
4. Utilizes tensor parallelism for multi-GPU scaling

### Configuring vLLM

The following environment variables can be set to configure vLLM behavior:

```bash
OLLAMA_USE_VLLM=true                      # Enable vLLM backend
OLLAMA_VLLM_GPU_MEMORY_UTILIZATION=0.8    # Memory utilization (0.0-1.0)
OLLAMA_VLLM_MAX_MODEL_LEN=8192            # Max context length
```

These settings are automatically configured when using the `start_vllm.sh` script.

### Kaggle Integration

This setup can be used in conjunction with the Kaggle notebook at <https://www.kaggle.com/code/vinayranavpcs/ollama-qwen3-custom> for publishing and deploying your application.