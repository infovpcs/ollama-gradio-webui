# Ollama Gradio WebUI

This project provides a user-friendly chat interface using Gradio for interacting with various Ollama models, with support for React agent integration specifically designed for Odoo development.

## Features

* Chat with multiple Ollama models (including Qwen2.5:14b)
* React Agent API for Odoo development assistance
* Visual Assistant for image analysis
* Multiple prompt templates for different use cases
* Web-based interface with public sharing capability

## How to Run

1. Install [Ollama](https://github.com/ollama/ollama)
2. Create conda environment: `conda create -n ollama python=3.12`
3. Activate conda environment: `conda activate ollama`
4. Install required packages: `pip install -r requirements.txt`
5. Run the application: `python app.py`

## React Agent Integration

The Ollama Gradio WebUI includes a dedicated React Agent API for Odoo development. This agent follows the Reason-Act-Observe pattern to help users generate high-quality Odoo modules based on their requirements.

API endpoints are available at:
* `/chat` - For direct chat interactions with any model
* `/react_agent` - For Odoo development assistance with the React agent