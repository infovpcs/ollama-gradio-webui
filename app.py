import gradio as gr
import ollama
import json
import base64
import copy
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to get models from Ollama API, with error handling
try:
    model_list = ollama.list()
    model_names = [model['model'] for model in model_list['models']]
    logger.info(f"Found {len(model_names)} models: {', '.join(model_names)}")
except Exception as e:
    logger.warning(f"Could not connect to Ollama API: {str(e)}")
    logger.info("Using default model list")
    model_names = ["qwen2.5:14b", "llama3:8b", "llava:7b-v1.6", "mistral:7b", "phi3:14b"]

# Add Qwen2.5:14b to the model list if not already present
if "qwen2.5:14b" not in model_names:
    model_names.append("qwen2.5:14b")
    logger.info("Added qwen2.5:14b to model list")
PROMPT_LIST = []
VL_CHAT_LIST = []

# Add React agent prompt
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
    logger.error(f"Error loading prompt.json: {str(e)}")
    PROMPT_DICT = {
        "React Agent": REACT_AGENT_PROMPT,
        "Translation Assistant": "You are a helpful translation assistant. Please translate my Chinese to English, and non-Chinese to Chinese.",
        "Code Assistant": "You are a professional programming assistant, please help me solve coding problems."
    }
    PROMPT_LIST = list(PROMPT_DICT.keys())
    logger.info("Using default prompts due to error loading prompt.json")
# Initialization function
def init():
    VL_CHAT_LIST.clear()
# Check if string contains Chinese characters
def contains_chinese(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fa5':
            return True
    return False
def ollama_chat(message, history, model_name, history_flag):
    """
    Main chat function for Ollama models.
    
    Args:
        message: User message
        history: Chat history
        model_name: Name of the Ollama model to use
        history_flag: Whether to include chat history
        
    Returns:
        The updated chat history with new message pairs
    """
    try:
        logger.info(f"Chat request for model: {model_name}")
        ollama_messages = []
        chat_message = {
            'role': 'user', 
            'content': message
        }
        
        # Include chat history if enabled
        if history_flag and len(history) > 0:
            for element in history:  
                # Skip None entries that might be in history from image processing
                if element[0] is not None:
                    history_user_message = {
                        'role': 'user', 
                        'content': element[0]
                    }
                    ollama_messages.append(history_user_message)
                    
                if element[1] is not None:
                    history_assistant_message = {
                        'role': 'assistant', 
                        'content': element[1]
                    }
                    ollama_messages.append(history_assistant_message)   
        
        ollama_messages.append(chat_message)
        logger.info(f"Sending {len(ollama_messages)} messages to {model_name}")
        
        # First, add the user message to the chat history
        new_history = history + [(message, None)]
        
        # Stream the response from Ollama
        try:
            stream = ollama.chat(
                model=model_name,
                messages=ollama_messages,
                stream=True
            )
            
            partial_message = ""
            for chunk in stream:
                if chunk and 'message' in chunk and 'content' in chunk['message'] and len(chunk['message']['content']) != 0:
                    partial_message = partial_message + chunk['message']['content']
                    # Update the last response in history
                    new_history[-1] = (message, partial_message)
                    yield new_history
        except Exception as e:
            # Handle streaming error
            error_message = f"Error: {str(e)}"
            logger.error(f"Streaming error in ollama_chat: {str(e)}")
            # Update the last response with the error message
            new_history[-1] = (message, error_message)
            yield new_history
            
    except Exception as e:
        # Handle overall function error
        error_message = f"Error: {str(e)}"
        logger.error(f"General error in ollama_chat: {str(e)}")
        # Create a properly formatted history with the error message
        new_history = history + [(message, error_message)]
        yield new_history
# Agent generation
def ollama_prompt(message, history, model_name, prompt_info):
    """
    Chat function for Ollama models with specific prompt templates.
    
    Args:
        message: User message
        history: Chat history
        model_name: Name of the Ollama model to use
        prompt_info: The prompt template to use
        
    Returns:
        The updated chat history with new message pairs
    """
    try:
        logger.info(f"Prompt request for model: {model_name}, prompt: {prompt_info}")
        
        # Prepare messages for Ollama API
        ollama_messages = []
        system_message = {
            'role': 'system', 
            'content': PROMPT_DICT[prompt_info]
        }
        user_message = {
            'role': 'user', 
            'content': message
        }
        ollama_messages.append(system_message)
        ollama_messages.append(user_message)
        
        # First, add the user message to the chat history
        new_history = history + [(message, None)]
        
        # Stream the response from Ollama
        try:
            stream = ollama.chat(
                model=model_name,
                messages=ollama_messages,
                stream=True
            )
            
            partial_message = ""
            for chunk in stream:
                if chunk and 'message' in chunk and 'content' in chunk['message'] and len(chunk['message']['content']) != 0:
                    partial_message = partial_message + chunk['message']['content']
                    # Update the last response in history
                    new_history[-1] = (message, partial_message)
                    yield new_history
        except Exception as e:
            # Handle streaming error
            error_message = f"Error: {str(e)}"
            logger.error(f"Streaming error in ollama_prompt: {str(e)}")
            # Update the last response with the error message
            new_history[-1] = (message, error_message)
            yield new_history
            
    except Exception as e:
        # Handle overall function error
        error_message = f"Error: {str(e)}"
        logger.error(f"General error in ollama_prompt: {str(e)}")
        # Create a properly formatted history with the error message
        new_history = history + [(message, error_message)]
        yield new_history
# Image upload
def vl_image_upload(image_path,chat_history):
    messsage = {
        "type":"image",
        "content":image_path
    }
    chat_history.append(((image_path,),None))
    VL_CHAT_LIST.append(messsage)
    return None,chat_history
# Submit question
def vl_submit_message(message,chat_history):
    messsage = {
        "type":"user",
        "content":message
    }
    chat_history.append((message,None))
    VL_CHAT_LIST.append(messsage)
    return "",chat_history
# Retry
def vl_retry(chat_history):
    if len(VL_CHAT_LIST)>1:
        if VL_CHAT_LIST[len(VL_CHAT_LIST)-1]['type'] == "assistant":
            VL_CHAT_LIST.pop()
            chat_history.pop()
    return chat_history
# Undo
def vl_undo(chat_history):
    message = ""
    chat_list = copy.deepcopy(VL_CHAT_LIST)
    if len(chat_list)>1:
        if chat_list[len(chat_list)-1]['type'] == "assistant":
            message = chat_list[len(chat_list)-2]['content']
            VL_CHAT_LIST.pop()
            VL_CHAT_LIST.pop()
            chat_history.pop()
            chat_history.pop()
        elif chat_list[len(chat_list)-1]['type'] == "user":
            message = chat_list[len(chat_list)-1]['content']
            VL_CHAT_LIST.pop()
            chat_history.pop()
    return message,chat_history
# Clear
def vl_clear():
    VL_CHAT_LIST.clear()
    return None,"",[]
# Return answer
def vl_submit(history_flag, chinese_flag, chat_history):
    """
    Submit function for the Visual Assistant tab.
    Processes the current chat history with the LLaVA model.
    
    Args:
        history_flag: Whether to use context
        chinese_flag: Whether to force Chinese output
        chat_history: Current chat history
        
    Returns:
        Updated chat history with the model's response
    """
    try:
        if len(VL_CHAT_LIST) > 1:
            try:
                # Get formatted messages for the model
                messages = get_vl_message(history_flag, chinese_flag)
                
                # Call the Ollama API
                response = ollama.chat(
                    model = "llava:7b-v1.6",
                    messages = messages
                )
                
                # Process the response
                result = response["message"]["content"]
                output = {
                    "type": "assistant",
                    "content": result
                }
                
                # Update the chat history with the result
                # For Gradio chatbot, we need to provide list of tuples (user, assistant)
                chat_history.append((None, result))
                VL_CHAT_LIST.append(output)
                
            except Exception as e:
                # Handle errors from the API call
                error_message = f"Error: {str(e)}"
                logger.error(f"Error in vl_submit API call: {str(e)}")
                
                # Add the error message to chat history in the correct format
                chat_history.append((None, error_message))
                
                # Show a warning in the UI
                gr.Warning(error_message)
        else:
            # Not enough chat messages yet
            message = "Please upload an image and add a message first."
            gr.Warning(message)
            # We don't modify the chat history in this case
    except Exception as e:
        # Handle any other unexpected errors
        error_message = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in vl_submit: {str(e)}")
        
        # Add the error message to chat history in the correct format
        chat_history.append((None, error_message))
        
        # Show a warning in the UI
        gr.Warning(error_message)
    
    # Always return the chat history, whether modified or not
    return chat_history
def get_vl_message(history_flag,chinese_flag):
    messages = []
    if history_flag:
        i=0
        while i<len(VL_CHAT_LIST):
            if VL_CHAT_LIST[i]['type']=="image" and VL_CHAT_LIST[i+1]['type']=="user":
                image_path = VL_CHAT_LIST[i]["content"]
                # Read the binary data of the image file
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                # Convert binary data to base64 encoded string
                base64_string = base64.b64encode(image_data).decode("utf-8")
                content = VL_CHAT_LIST[i+1]["content"]
                chat_message = {
                    'role': 'user', 
                    'content': content,
                    'images':[base64_string]
                }
                messages.append(chat_message)
                i+=2
            elif VL_CHAT_LIST[i]['type']=="assistant":
                assistant_message = {
                    "role":"assistant",
                    "content":VL_CHAT_LIST[i]['content']
                }
                messages.append(assistant_message)
                i+=1
            elif VL_CHAT_LIST[i]['type']=="user":
                user_message = {
                    "role":"user",
                    "content":VL_CHAT_LIST[i]['content']
                }
                messages.append(user_message)
                i+=1
            else:
                i+=1
    else:
        if VL_CHAT_LIST[0]['type']=="image" and VL_CHAT_LIST[-1]['type']=="user":
            image_path = VL_CHAT_LIST[0]["content"]
            # Read the binary data of the image file
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            # Convert binary data to base64 encoded string
            base64_string = base64.b64encode(image_data).decode("utf-8")
            content = VL_CHAT_LIST[-1]["content"]
            chat_message = {
                'role': 'user', 
                'content': content,
                'images':[base64_string]
            }
            messages.append(chat_message)
    if chinese_flag:
        system_message = {
            'role': 'system', 
            'content': 'You are a Helpful Assistant. Please answer the question in Chinese.'
        }
        messages.insert(0,system_message)
    return messages
with gr.Blocks(title="Ollama WebUI", fill_height=True) as demo:
    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=1):
                model_info = gr.Dropdown(model_names, value="", allow_custom_value=True, label="Model Selection")
                history_flag = gr.Checkbox(label="Enable Context")
            with gr.Column(scale=4):
                chat_bot = gr.Chatbot(height=600)
                with gr.Row():
                    with gr.Column(scale=4):
                        text_box = gr.Textbox(label="Message", placeholder="Enter message...", lines=2)
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("Submit", variant="primary")
                        retry_btn = gr.Button("🔄 Retry")
                        undo_btn = gr.Button("↩️ Undo")
                        clear_btn = gr.Button("🗑️ Clear")
                
                # Set up event handlers
                submit_btn.click(
                    fn=ollama_chat,
                    inputs=[text_box, chat_bot, model_info, history_flag],
                    outputs=[chat_bot],
                    api_name="chat"
                )
                retry_btn.click(fn=lambda: None)  # TODO: Implement retry functionality
                undo_btn.click(fn=lambda: None)   # TODO: Implement undo functionality
                clear_btn.click(fn=lambda: [None, []])  # Clear chat history
    with gr.Tab("Agent"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt_model_info = gr.Dropdown(model_names, value="", allow_custom_value=True, label="Model Selection")
                prompt_info = gr.Dropdown(choices=PROMPT_LIST,value=PROMPT_LIST[0],label="Agent Selection",interactive=True)
            with gr.Column(scale=4):
                prompt_chat_bot = gr.Chatbot(height=600)
                with gr.Row():
                    with gr.Column(scale=4):
                        prompt_text_box = gr.Textbox(label="Message", placeholder="Enter message...", lines=2)
                    with gr.Column(scale=1):
                        prompt_submit_btn = gr.Button("Submit", variant="primary")
                        prompt_retry_btn = gr.Button("🔄 Retry")
                        prompt_undo_btn = gr.Button("↩️ Undo")
                        prompt_clear_btn = gr.Button("🗑️ Clear")
                
                # Set up event handlers
                prompt_submit_btn.click(
                    fn=ollama_prompt,
                    inputs=[prompt_text_box, prompt_chat_bot, prompt_model_info, prompt_info],
                    outputs=[prompt_chat_bot],
                    api_name="prompt"
                )
                prompt_retry_btn.click(fn=lambda: None)  # TODO: Implement retry functionality
                prompt_undo_btn.click(fn=lambda: None)   # TODO: Implement undo functionality
                prompt_clear_btn.click(fn=lambda: [None, []])  # Clear chat history
    with gr.Tab("Visual Assistant"):
        with gr.Row():
            with gr.Column(scale=1):
                history_flag = gr.Checkbox(label="Enable Context")
                chinese_flag = gr.Checkbox(value=True,label="Force Chinese Output")
                image = gr.Image(type="filepath")
            with gr.Column(scale=4):
                chat_bot = gr.Chatbot(height=600)
                with gr.Row():
                    retry_btn = gr.Button("🔄 Retry")
                    undo_btn = gr.Button("↩️ Undo")
                    clear_btn = gr.Button("🗑️ Clear")
                with gr.Row():
                    message = gr.Textbox(show_label=False,container=False,scale=5)
                    submit_btn = gr.Button("Submit",variant="primary",scale=1)
        image.upload(fn=vl_image_upload,inputs=[image,chat_bot],outputs=[image,chat_bot])
        submit_btn.click(fn=vl_submit_message,inputs=[message,chat_bot],outputs=[message,chat_bot]).then(fn=vl_submit,inputs=[history_flag,chinese_flag,chat_bot],outputs=[chat_bot])
        retry_btn.click(fn=vl_retry,inputs=[chat_bot],outputs=[chat_bot]).then(fn=vl_submit,inputs=[history_flag,chinese_flag,chat_bot],outputs=[chat_bot])
        undo_btn.click(fn=vl_undo,inputs=[chat_bot],outputs=[message,chat_bot])
        clear_btn.click(fn=vl_clear,inputs=[],outputs=[image,message,chat_bot])
    demo.load(fn=init)
# Define API functions that will be exposed through the Gradio interface

def api_chat(message, model_name="qwen2.5:14b", enable_context=True):
    """
    API endpoint for direct model interaction.
    
    Args:
        message: The prompt to send to the model
        model_name: The model to use (default: qwen2.5:14b)
        enable_context: Whether to enable context preservation
        
    Returns:
        The generated response
    """
    try:
        logger.info(f"API request: model={model_name}, message={message[:50]}...")
        
        # Process request directly without streaming
        messages = [{
            'role': 'user',
            'content': message
        }]
        
        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        
        result = response["message"]["content"]
        logger.info(f"Generated response (first 50 chars): {result[:50]}...")
        return result
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(f"API error: {str(e)}")
        # For Gradio Textbox output, returning a string is fine
        return error_message

def api_react_agent(query, odoo_version="18.0", model_name="qwen2.5:14b"):
    """
    API endpoint specifically for the React agent.
    
    Args:
        query: The user's query about Odoo development
        odoo_version: The Odoo version to target
        model_name: The model to use
        
    Returns:
        The generated response as a string (for use with Textbox components)
    """
    try:
        logger.info(f"React agent request: model={model_name}, odoo_version={odoo_version}, query={query[:50]}...")
        
        # Create a prompt for Odoo code generation
        system_message = {
            'role': 'system',
            'content': f"""You are an expert Odoo developer assistant. Your task is to help generate high-quality Odoo {odoo_version} code based on user requirements.
            Provide clear, well-structured code with proper docstrings and comments.
            Follow Odoo coding standards and best practices.
            When showing code, explain key parts of the implementation."""
        }
        
        user_message = {
            'role': 'user',
            'content': query
        }
        
        messages = [system_message, user_message]
        
        # Make the API call with proper error handling
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages
            )
            
            # Extract and return the content from the response
            if response and "message" in response and "content" in response["message"]:
                result = response["message"]["content"]
                logger.info(f"Generated React agent response (first 50 chars): {result[:50]}...")
                return result
            else:
                # Handle unexpected response format
                logger.error(f"Unexpected response format from Ollama API: {response}")
                return "Error: Unexpected response format from model API"
                
        except Exception as e:
            # Handle API call errors
            api_error = f"API call error: {str(e)}"
            logger.error(f"React agent API call error: {str(e)}")
            return api_error
            
    except Exception as e:
        # Handle overall function errors
        error_message = f"Error: {str(e)}"
        logger.error(f"React agent general error: {str(e)}")
        # For Gradio Textbox output, returning a string is fine
        return error_message

# Create API endpoints for the Gradio interface
with gr.Blocks(title="React Agent API", css=".footer {display:none}") as api_interface:
    with gr.Tab("Direct Chat"):
        gr.Markdown("# Direct Chat API")
        with gr.Row():
            message_input = gr.Textbox(label="Message")
            model_dropdown = gr.Dropdown(choices=model_names, value="qwen2.5:14b", label="Model")
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
                value="qwen2.5:14b", 
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
    ["Ollama WebUI", "React Agent API"]
) as app:
    pass

if __name__ == "__main__":
    logger.info("Starting Ollama WebUI with sharing enabled...")
    app.launch(share=True)