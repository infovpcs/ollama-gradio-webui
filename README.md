# Ollama Gradio WebUI with vLLM

This project provides a user-friendly chat interface using Gradio for interacting with various Ollama models, with special support for Qwen3 models using vLLM acceleration and React agent integration specifically designed for Odoo development. The vLLM backend provides 2-4x faster inference and extended context window capabilities.

![Ollama Gradio WebUI](https://www.vperfectcs.com/web/image/website/1/logo/vpcsinfo?unique=eabe3d6)

## Features

* **vLLM Acceleration**: 2-4x faster inference compared to standard Ollama models
* **Extended Context Window**: Up to 16K tokens for Qwen3:vpcs model for handling longer conversations
* **Performance Tracking**: Real-time statistics including tokens/second and response times
* **React Agent for Odoo**: Specialized AI assistant for Odoo module development following the React pattern
* **Multi-Environment Support**: Run locally or in cloud environments like Kaggle with automatic optimization
* **Visual Assistant**: Image analysis capabilities with LLaVA model integration
* **Multilingual Support**: Built-in Hindi translation functionality
* **Template System**: Customizable prompt templates for different use cases
* **File Upload**: Process documents and code files for context-enriched responses
* **Public Sharing**: Generate temporary public URLs for collaborative sessions
* **Model Management**: Automatic model setup and configuration

## Applications

This repository contains three main applications, each suited for different use cases:

1. **app.py** - The full-featured application 
   * Complete UI with all capabilities
   * Chat, React agent, translation, and visual processing
   * Ideal for local use with standard Ollama

2. **vllm_app.py** - Enhanced performance version
   * vLLM optimization for 2-4x faster inference
   * Performance tracking and benchmarking
   * Extended context window support
   * Compatible with Kaggle for cloud deployment

3. **simple_chat.py** - Minimalist version
   * Focused solely on chat functionality
   * Lower resource requirements
   * Simplified interface for basic use cases

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

#### What Each Script Does:

**Standard Script (`start.sh`)**
- ✅ Detects your environment (macOS/Linux/Kaggle) automatically
- 🔍 Checks if Ollama server is running and starts it if needed
- 📦 Creates the Qwen3:vpcs model with extended context window (16K tokens)
- 🔧 Installs all required dependencies
- 📄 Sets up default prompt templates

**vLLM Script (`start_vllm.sh`)**
- 🚀 Everything in the standard script, plus:
- 🔋 Enables vLLM acceleration for 2-4x faster inference
- 🧠 Configures optimal GPU memory utilization (90% in Kaggle, 80% locally)
- 📏 Sets up extended context window support (16K tokens)
- 📊 Enables performance tracking and benchmarking
- 🔌 Auto-detects Kaggle environment and adapts configuration

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

## API Integration

The application provides API endpoints that can be exposed through ngrok for integration with external applications.

### Using the API Server

You can run the application as an API server using the provided `api_server.py` script:

```bash
# Install the required dependency
pip install pyngrok

# Start the API server with ngrok tunnel
python api_server.py --kaggle   # Add --kaggle flag if running in Kaggle
```

This will start the application and create an ngrok tunnel, providing you with a public URL for API access.

### Available API Endpoints

The following API endpoints are available:

- **Chat API**: `/api/chat`
  ```json
  {
    "message": "Your message here",
    "model": "qwen3:vpcs-vllm",
    "enable_context": true
  }
  ```

- **React Agent API**: `/api/react_agent`
  ```json
  {
    "query": "Create an Odoo module for inventory management",
    "odoo_version": "18.0",
    "model": "qwen3:vpcs-vllm"
  }
  ```

### Integration in Kaggle

You can integrate with external applications directly from Kaggle using ngrok:

```python
from pyngrok import ngrok

# Create a tunnel to the Gradio app
listener = ngrok.connect(addr="localhost:7860")
print(f"API accessible at: {listener.public_url}")
```

A complete example notebook `kaggle_api_template.ipynb` is provided in the repository.

## Kaggle Integration

This application is fully compatible with Kaggle Notebooks, allowing you to leverage Kaggle's free GPU resources for faster inference.

### Running in Kaggle

You can run the application directly in a Kaggle notebook with these simple steps:

```python
# Clone the repository
!git clone https://github.com/infovpcs/ollama-gradio-webui.git
%cd ollama-gradio-webui

# Method 1: Using the provided scripts
!sh install_deps.sh
!sh start_vllm.sh

# Alternative Method 2: Direct execution
import os

# Set required environment variables
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["OLLAMA_USE_VLLM"] = "true"
os.environ["KAGGLE_API_MODE"] = "true"

# Install core dependencies if needed
!pip install -q gradio==3.50.2 ollama==0.1.6 requests markdown

# Launch the app
!python vllm_app.py --kaggle-mode
```

### Benefits of Kaggle Integration

- **Free GPU Access**: Utilize Kaggle's T4 GPUs for faster inference
- **Public Sharing**: Generate a shareable URL that remains active for 72 hours
- **No Setup Required**: No need to install Ollama locally
- **Optimized Configuration**: Automatically configures for optimal performance in Kaggle's environment

### Example Notebook

A complete example notebook is available at: [Ollama Qwen3 Custom with vLLM](https://www.kaggle.com/code/vinayranavpcs/ollama-qwen3-custom)

### Troubleshooting Kaggle Integration

**Common Issues & Solutions:**

1. **Internet Access Required**
   - Ensure "Internet" is enabled in Kaggle notebook settings
   - Found under "Settings" > "Internet" > "Internet connected: On"

2. **Warning Messages**
   - `"invalid option provided" option=gpu_memory_utilization`: Safe to ignore, these are vLLM-specific parameters
   - `"deps.sh: [line number]: [[: not found"`: Shell syntax compatibility issue, but doesn't affect functionality
   - Gradio version notices can be safely ignored

3. **GPU Not Detected**
   - Make sure the notebook has GPU acceleration enabled
   - Select "Settings" > "Accelerator" > "GPU T4 x1"

4. **Application Not Responding**
   - Check if another instance of Ollama is already running
   - Try a restart of the Kaggle notebook's runtime

5. **URL Access**
   - Kaggle generates a public URL valid for 72 hours
   - You may need to scroll to find it in the output logs