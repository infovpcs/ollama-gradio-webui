"""
Ollama Gradio WebUI with vLLM Optimization for Qwen3:vpcs-vllm

This application provides a Gradio web interface for interacting with locally running
Ollama models, with specific optimizations for vLLM-backed models like Qwen3:vpcs-vllm.

Features:
- Chat interface with conversation history
- File uploads for context
- Performance metrics tracking
- Model parameter customization
- Streaming responses
- React agent mode for Odoo development
- Extended context handling
"""

import gradio as gr
import ollama
import json
import base64
import copy
import os
import sys
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Union
import markdown
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration for vLLM
VLLM_CONFIG = {
    "enabled": True,  # Set to False to disable vLLM optimizations
    "gpu_memory_utilization": 0.8,
    "max_model_len": 8192,
    "tensor_parallel_size": 1,  # Set to number of GPUs if using multiple GPUs
}

# Check if Ollama is configured for vLLM
def check_vllm_status():
    """Check if Ollama is configured to use vLLM backend"""
    try:
        # Simple test query to check environment
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:vpcs-vllm",
                "messages": [
                    {"role": "user", "content": "Are you running on vLLM backend? Reply with only yes or no."}
                ],
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('message', {}).get('content', '').lower()
            
            # Check if the response indicates vLLM is being used
            if 'yes' in response_text and ('vllm' in response_text or 'backend' in response_text):
                logger.info("✅ vLLM backend detected")
                return True
            else:
                logger.info("⚠️ vLLM backend not detected in response")
                return False
        else:
            logger.warning(f"❌ Error checking vLLM status: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"❌ Could not connect to Ollama API: {str(e)}")
        return False

# Try to get models from Ollama API, with error handling
try:
    model_list = ollama.list()
    model_names = [model['model'] for model in model_list['models']]
    logger.info(f"Found {len(model_names)} models: {', '.join(model_names)}")
except Exception as e:
    logger.warning(f"Could not connect to Ollama API: {str(e)}")
    logger.info("Using default model list")
    model_names = ["qwen3:vpcs-vllm", "qwen3:latest", "qwen2.5:14b", "llama3:8b", "llava:7b-v1.6", "mistral:7b", "phi3:14b"]

# Add Qwen models to the model list if not already present
for model in ["qwen3:vpcs-vllm", "qwen3:latest", "qwen2.5:14b"]:
    if model not in model_names:
        model_names.append(model)
        logger.info(f"Added {model} to model list")

# Check if qwen3:vpcs-vllm is available and make it the default if so
default_model = "qwen3:vpcs-vllm" if "qwen3:vpcs-vllm" in model_names else model_names[0]
logger.info(f"Using {default_model} as default model")

# Model parameters
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 42,
    "num_predict": 512,
}

# Load prompts
PROMPT_LIST = []
VL_CHAT_LIST = []

# Add React agent prompt with Odoo focus
REACT_AGENT_PROMPT = """You are an expert Odoo developer assistant that follows the REACT pattern:
1. Reason about what needs to be done
2. Choose an Action to take
3. Observe the result and continue reasoning

Your goal is to help users generate high-quality Odoo modules based on their requirements.
When responding to the user, be concise, clear and professional.
"""

# Parse prompts
try:
    with open("prompt.json", "r", encoding="utf-8") as f:
        PROMPT_DICT = json.load(f)
        for key in PROMPT_DICT:
            PROMPT_LIST.append(key)
    
    # Add React Agent to the prompt list if not already present
    if "React Agent" not in PROMPT_DICT:
        PROMPT_DICT["React Agent"] = REACT_AGENT_PROMPT
        PROMPT_LIST.append("React Agent")
        logger.info("Added React Agent to prompt list")
        
        # Save updated prompt.json
        with open("prompt.json", "w", encoding="utf-8") as f:
            json.dump(PROMPT_DICT, f, ensure_ascii=False, indent=4)
except Exception as e:
    logger.error(f"Error loading prompts: {str(e)}")
    PROMPT_DICT = {"React Agent": REACT_AGENT_PROMPT}
    PROMPT_LIST = ["React Agent"]

# Performance tracking
class PerformanceTracker:
    """Track performance metrics for model responses"""
    def __init__(self):
        self.metrics = []
        
    def track(self, model: str, prompt_length: int, response_length: int, time_taken: float):
        """Track a new performance metric"""
        self.metrics.append({
            "model": model,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "time_taken": time_taken,
            "tokens_per_second": response_length / time_taken if time_taken > 0 else 0,
            "timestamp": time.time()
        })
        
        # Keep only the last 100 metrics
        if len(self.metrics) > 100:
            self.metrics.pop(0)
    
    def get_model_stats(self, model: str) -> Dict[str, Any]:
        """Get performance statistics for a specific model"""
        model_metrics = [m for m in self.metrics if m["model"] == model]
        if not model_metrics:
            return {"average_tokens_per_second": 0, "count": 0}
        
        avg_tps = sum(m["tokens_per_second"] for m in model_metrics) / len(model_metrics)
        return {
            "average_tokens_per_second": avg_tps,
            "count": len(model_metrics)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all models"""
        models = set(m["model"] for m in self.metrics)
        return {model: self.get_model_stats(model) for model in models}

# Initialize performance tracker
performance_tracker = PerformanceTracker()

# Direct API interaction for more control
def api_direct_request(model: str, messages: List[Dict[str, str]], 
                       params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a direct API request to Ollama with specified parameters"""
    if params is None:
        params = {}
    
    # Merge with default parameters
    request_params = copy.deepcopy(DEFAULT_PARAMS)
    request_params.update(params)
    
    # Prepare the payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        **request_params
    }
    
    # Add vLLM parameters if enabled
    if VLLM_CONFIG["enabled"]:
        payload["options"] = {
            "gpu_memory_utilization": VLLM_CONFIG["gpu_memory_utilization"],
            "max_model_len": VLLM_CONFIG["max_model_len"]
        }
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            elapsed_time = time.time() - start_time
            
            # Track performance
            prompt_text = "\n".join([m["content"] for m in messages])
            response_text = result.get('message', {}).get('content', '')
            performance_tracker.track(
                model, 
                len(prompt_text), 
                len(response_text), 
                elapsed_time
            )
            
            logger.info(f"Generated {len(response_text)} chars in {elapsed_time:.2f}s ({len(response_text)/elapsed_time:.2f} chars/s)")
            return result
        else:
            logger.error(f"API request failed: {response.status_code}")
            return {"message": {"content": f"Error: API request failed with status {response.status_code}"}}
    except Exception as e:
        logger.error(f"API request exception: {str(e)}")
        return {"message": {"content": f"Error: {str(e)}"}}

# Chat function
def chat(message, history, system_prompt, model, temperature, top_p, max_length, file=None):
    """Chat function with file upload support and performance tracking"""
    try:
        # Process file if provided
        file_content = ""
        if file is not None:
            try:
                file_type = file.name.split(".")[-1].lower()
                if file_type in ["txt", "py", "json", "md", "html", "css", "js", "xml"]:
                    file_content = file.read().decode("utf-8")
                    message = f"Here's a file I want you to analyze:\n\n```{file_type}\n{file_content}\n```\n\n{message}"
                else:
                    with open(file.name, "rb") as f:
                        file_bytes = f.read()
                    file_b64 = base64.b64encode(file_bytes).decode("utf-8")
                    message = f"[File: {file.name} (base64 encoded)]({file_b64})\n{message}"
            except Exception as e:
                message = f"[Error reading file: {str(e)}]\n{message}"
        
        # Convert history to Ollama format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        messages.append({"role": "user", "content": message})
        
        # Parameters for the model
        params = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_length)
        }
        
        # Call API directly for more control
        start_time = time.time()
        response = api_direct_request(model, messages, params)
        elapsed_time = time.time() - start_time
        
        # Extract and return response
        response_text = response.get('message', {}).get('content', '')
        
        # Add performance info
        performance_info = f"\n\n---\n*Generated {len(response_text)} chars in {elapsed_time:.2f}s ({len(response_text)/elapsed_time:.2f} chars/s)*"
        
        # Return response in the format expected by Gradio chatbot (list of lists)
        # Append the new message and response to the history
        history = history + [(message, response_text)]
        return history
    except Exception as e:
        error_message = f"Error: {str(e)}\n\nThis is likely due to a connection issue with Ollama API."
        # Return in the format expected by Gradio chatbot
        history = history + [(message, error_message)]
        return history

# React agent chat
def react_agent_chat(message, history, odoo_version, model, temperature, max_length):
    """Chat function specifically for the React agent pattern"""
    try:
        # Create system prompt for React agent
        system_prompt = f"""You are an expert Odoo {odoo_version} developer assistant that follows the REACT pattern:
1. Reason about what needs to be done
2. Choose an Action to take
3. Observe the result and continue reasoning

Your goal is to help users generate high-quality Odoo modules based on their requirements.
When responding to the user, be concise, clear and professional.
"""
        
        # Convert history to Ollama format
        messages = [{"role": "system", "content": system_prompt}]
        
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        messages.append({"role": "user", "content": message})
        
        # Parameters for the model
        params = {
            "temperature": float(temperature),
            "num_predict": int(max_length)
        }
        
        # Call API directly
        start_time = time.time()
        response = api_direct_request(model, messages, params)
        elapsed_time = time.time() - start_time
        
        # Extract and return response
        response_text = response.get('message', {}).get('content', '')
        
        # Add performance info
        performance_info = f"\n\n---\n*Generated {len(response_text)} chars in {elapsed_time:.2f}s ({len(response_text)/elapsed_time:.2f} chars/s)*"
        
        # Return response in the format expected by Gradio chatbot (list of lists)
        # Append the new message and response to the history
        history = history + [(message, response_text)]
        return history
    except Exception as e:
        error_message = f"Error: {str(e)}\n\nThis is likely due to a connection issue with Ollama API."
        # Return in the format expected by Gradio chatbot
        history = history + [(message, error_message)]
        return history

# API chat endpoint
def api_chat(message, model, enable_context=True, system_prompt=""):
    """Simple API endpoint for chat"""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if not enable_context:
            # Just use the current message without context
            messages.append({"role": "user", "content": message})
        else:
            # We would add context here if needed
            messages.append({"role": "user", "content": message})
        
        response = api_direct_request(model, messages)
        return response.get('message', {}).get('content', '')
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message

# React agent API endpoint
def api_react_agent(query, odoo_version, model):
    """API endpoint for React agent"""
    try:
        system_prompt = f"""You are an expert Odoo {odoo_version} developer assistant that follows the REACT pattern:
1. Reason about what needs to be done
2. Choose an Action to take
3. Observe the result and continue reasoning

Your goal is to help users generate high-quality Odoo modules based on their requirements.
When responding to the user, be concise, clear and professional.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = api_direct_request(model, messages)
        return response.get('message', {}).get('content', '')
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message

# Get vLLM status
vllm_enabled = check_vllm_status()
vllm_status = "✅ Enabled" if vllm_enabled else "❌ Not detected"

# Create main Gradio interface
with gr.Blocks(title="Ollama WebUI with vLLM", css=".footer {display:none} .chatbot {height: 600px !important; overflow-y: auto;}") as demo:
    gr.Markdown(f"""
    # 🚀 Ollama WebUI with vLLM Optimization
    
    This interface is optimized for high-performance chat with vLLM-backed Ollama models.
    
    - **vLLM Status**: {vllm_status}
    - **Default Model**: {default_model}
    - **Max Context Length**: {VLLM_CONFIG['max_model_len']} tokens
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                with gr.Column(scale=8):
                    message = gr.Textbox(
                        placeholder="Enter your message here...",
                        lines=5,
                    )
                with gr.Column(scale=1):
                    submit = gr.Button("Send")
            with gr.Row():
                clear = gr.Button("Clear")
                file_upload = gr.File(label="Upload File for Context")
        
        with gr.Column(scale=1):
            model = gr.Dropdown(choices=model_names, value=default_model, label="Model")
            system_prompt = gr.Textbox(
                placeholder="Enter a system prompt (optional)",
                label="System Prompt",
                lines=3,
            )
            with gr.Accordion("Prompt Templates", open=False):
                prompt_template = gr.Dropdown(choices=PROMPT_LIST, value="React Agent", label="Template")
                load_prompt = gr.Button("Load")
            
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(minimum=0.01, maximum=2.0, value=0.7, step=0.01, label="Temperature")
                top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.9, step=0.01, label="Top P")
                max_length = gr.Slider(minimum=32, maximum=4096, value=512, step=32, label="Max Length")
                model_info = gr.Markdown("*Model Info: Select a model to see details*")
            
            with gr.Accordion("Performance", open=False):
                performance_info = gr.Markdown("*No performance data available yet*")
                refresh_performance = gr.Button("Refresh Stats")

    # Handle prompt template loading
    def load_prompt_template(template_name):
        if template_name in PROMPT_DICT:
            return PROMPT_DICT[template_name]
        return ""
    
    load_prompt.click(
        fn=load_prompt_template,
        inputs=prompt_template,
        outputs=system_prompt
    )
    
    # Handle message submission
    submit_event = submit.click(
        fn=chat,
        inputs=[message, chatbot, system_prompt, model, temperature, top_p, max_length, file_upload],
        outputs=[chatbot],
        queue=True,
    ).then(
        fn=lambda: "",
        outputs=message,
    ).then(
        fn=lambda: None,
        outputs=file_upload,
    )
    
    message.submit(
        fn=chat,
        inputs=[message, chatbot, system_prompt, model, temperature, top_p, max_length, file_upload],
        outputs=[chatbot],
        queue=True,
    ).then(
        fn=lambda: "",
        outputs=message,
    ).then(
        fn=lambda: None,
        outputs=file_upload,
    )
    
    # Clear chat
    clear.click(lambda: None, None, chatbot)
    
    # Update model info
    def update_model_info(model_name):
        try:
            model_info = ollama.show(model_name)
            parameters = model_info.get('parameters', {})
            
            # Format parameters for display
            param_str = "\n".join([f"- **{k}**: {v}" for k, v in parameters.items()])
            
            # Get performance stats if available
            stats = performance_tracker.get_model_stats(model_name)
            perf_str = f"- **Average speed**: {stats['average_tokens_per_second']:.2f} tokens/s\n- **Queries**: {stats['count']}"
            
            return f"""### Model: {model_name}
            
{param_str}

### Performance:
{perf_str}
            """
        except Exception as e:
            return f"*Error getting model info: {str(e)}*"
    
    model.change(
        fn=update_model_info,
        inputs=model,
        outputs=model_info
    )
    
    # Update performance info
    def update_performance_display():
        stats = performance_tracker.get_all_stats()
        if not stats:
            return "*No performance data available yet*"
        
        result = "### Performance Statistics\n\n"
        for model, model_stats in stats.items():
            result += f"**{model}**: {model_stats['average_tokens_per_second']:.2f} tokens/s ({model_stats['count']} queries)\n\n"
        
        return result
    
    refresh_performance.click(
        fn=update_performance_display,
        outputs=performance_info
    )
    
    # React Agent tab
    with gr.Tab("React Agent"):
        gr.Markdown("""
        # 🤖 React Agent for Odoo Development
        
        This specialized interface follows the React pattern for Odoo development:
        1. **Reason** about what needs to be done
        2. Choose an **Action** to take
        3. **Observe** the result and continue reasoning
        
        Perfect for developing Odoo modules and technical assistance.
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                agent_chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    with gr.Column(scale=8):
                        agent_message = gr.Textbox(
                            placeholder="Enter your Odoo development question here...",
                            lines=5,
                        )
                    with gr.Column(scale=1):
                        agent_submit = gr.Button("Send")
                agent_clear = gr.Button("Clear")
            
            with gr.Column(scale=1):
                odoo_version = gr.Dropdown(
                    choices=["16.0", "17.0", "18.0"], 
                    value="18.0", 
                    label="Odoo Version"
                )
                agent_model = gr.Dropdown(
                    choices=model_names, 
                    value=default_model, 
                    label="Model"
                )
                agent_temperature = gr.Slider(
                    minimum=0.01, 
                    maximum=2.0, 
                    value=0.7, 
                    step=0.01, 
                    label="Temperature"
                )
                agent_max_length = gr.Slider(
                    minimum=32, 
                    maximum=4096, 
                    value=1024, 
                    step=32, 
                    label="Max Length"
                )
                
                with gr.Accordion("React Agent Info", open=False):
                    gr.Markdown("""
                    The React Agent pattern helps break down complex Odoo development tasks:
                    
                    1. **Reasoning**: Analyzing the requirements and thinking through the solution
                    2. **Action**: Deciding what code to write or modify
                    3. **Observation**: Evaluating the results and adjusting the approach
                    
                    This pattern is particularly effective for technical problem-solving in Odoo.
                    """)
        
        # Handle React agent message submission
        agent_submit.click(
            fn=react_agent_chat,
            inputs=[agent_message, agent_chatbot, odoo_version, agent_model, agent_temperature, agent_max_length],
            outputs=[agent_chatbot],
            queue=True,
        ).then(
            fn=lambda: "",
            outputs=agent_message,
        )
        
        agent_message.submit(
            fn=react_agent_chat,
            inputs=[agent_message, agent_chatbot, odoo_version, agent_model, agent_temperature, agent_max_length],
            outputs=[agent_chatbot],
            queue=True,
        ).then(
            fn=lambda: "",
            outputs=agent_message,
        )
        
        # Clear React agent chat
        agent_clear.click(lambda: None, None, agent_chatbot)

# Create API endpoints for the Gradio interface
with gr.Blocks(title="API Endpoints", css=".footer {display:none}") as api_interface:
    with gr.Tab("Direct Chat"):
        gr.Markdown("# Direct Chat API")
        with gr.Row():
            message_input = gr.Textbox(label="Message")
            model_dropdown = gr.Dropdown(choices=model_names, value=default_model, label="Model")
            context_checkbox = gr.Checkbox(label="Enable Context", value=True)
        response_output = gr.Textbox(label="Response")
        submit_button = gr.Button("Generate")
        submit_button.click(
            fn=api_chat,
            inputs=[message_input, model_dropdown, context_checkbox],
            outputs=response_output
        )
    
    with gr.Tab("React Agent"):
        gr.Markdown("# React Agent API")
        with gr.Row():
            query_input = gr.Textbox(label="Query")
            odoo_version_dropdown = gr.Dropdown(
                choices=["16.0", "17.0", "18.0"], 
                value="18.0", 
                label="Odoo Version"
            )
            api_model_dropdown = gr.Dropdown(
                choices=model_names, 
                value=default_model, 
                label="Model"
            )
        agent_output = gr.Textbox(label="Response")
        agent_button = gr.Button("Generate")
        agent_button.click(
            fn=api_react_agent,
            inputs=[query_input, odoo_version_dropdown, api_model_dropdown],
            outputs=agent_output
        )

# Main Gradio app interface with API endpoints included
with gr.TabbedInterface(
    [demo, api_interface],
    ["Ollama WebUI", "API Endpoints"]
) as app:
    pass

if __name__ == "__main__":
    # Display initial vLLM status
    if vllm_enabled:
        logger.info("🚀 vLLM backend detected - running with optimized performance")
    else:
        logger.info("⚠️ vLLM backend not detected - running in standard mode")
        logger.info("To enable vLLM, set environment variables: OLLAMA_USE_VLLM=true OLLAMA_VLLM_GPU_MEMORY_UTILIZATION=0.8")
    
    # Launch the app
    logger.info("Starting Ollama WebUI with sharing and queue enabled...")
    app.queue().launch(share=True)
