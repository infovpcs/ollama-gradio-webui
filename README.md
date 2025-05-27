# Ollama Gradio WebUI

This project provides a user-friendly chat interface using Gradio for interacting with various Ollama models, with special support for Qwen3 models and React agent integration specifically designed for Odoo development.

## Features

* Chat with multiple Ollama models (including Qwen3:vpcs with extended context window)
* Fixed interface with reliable response display across all models
* React Agent API for Odoo development assistance
* Visual Assistant for image analysis with LLaVA model
* Hindi translation capability for multilingual support
* Multiple prompt templates for different use cases
* Web-based interface with public sharing capability

## Applications

This repository contains two main applications:

1. **app.py** - The full-featured application with all capabilities (chat, React agent, translation, visual processing)
2. **simple_chat.py** - A simplified version focused only on chat functionality

## How to Run

### Using the Startup Script (Recommended)

The easiest way to run the application is using the provided startup script:

```bash
# Run the full application
./start.sh

# Run the simplified chat version
./start.sh -s
```

The startup script automatically:
- Checks if Ollama server is running and starts it if needed
- Ensures the Qwen3:vpcs model with extended context window is created
- Installs dependencies
- Creates a default prompt.json file if needed

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