import gradio as gr
import ollama
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
    model_names = ["qwen3:vpcs", "qwen3:latest", "qwen2.5:14b", "llama3:8b", "llava:7b-v1.6", "mistral:7b", "phi3:14b"]

# Add Qwen models to the model list if not already present
for model in ["qwen3:vpcs", "qwen3:latest", "qwen2.5:14b"]:
    if model not in model_names:
        model_names.append(model)
        logger.info(f"Added {model} to model list")

# Simplified chat function
def chat_with_model(message, chat_history, model_name):
    """Simple chat function that works reliably with Ollama models"""
    if message.strip() == "":
        return chat_history
    
    # Format history for Ollama API
    ollama_messages = []
    for user_msg, bot_msg in chat_history:
        ollama_messages.append({"role": "user", "content": user_msg})
        if bot_msg:  # Only add non-empty bot messages
            ollama_messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current message
    ollama_messages.append({"role": "user", "content": message})
    
    try:
        # Call Ollama API
        logger.info(f"Sending chat request to model: {model_name}")
        response = ollama.chat(
            model=model_name,
            messages=ollama_messages,
            stream=False
        )
        
        # Extract response content
        bot_response = response["message"]["content"]
        logger.info(f"Received response: {bot_response[:50]}...")
        
        # Update chat history
        chat_history.append((message, bot_response))
        return chat_history
    
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(f"Chat error: {error_message}")
        chat_history.append((message, error_message))
        return chat_history

# Function to check model context size
def check_context_size(model_name):
    try:
        # Simple prompt to check context size
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "What is your maximum context window size in tokens?"}]
        )
        return f"Model: {model_name}\nResponse: {response['message']['content']}"
    except Exception as e:
        return f"Error checking context size: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Ollama Chat with Qwen3") as demo:
    gr.Markdown("""
    # Ollama Chat Interface
    ## Using Qwen3 with Extended Context Window
    
    This simplified interface connects to Ollama for reliable chat functionality.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=model_names,
                value="qwen3:vpcs" if "qwen3:vpcs" in model_names else model_names[0],
                label="Model"
            )
            context_info = gr.Textbox(label="Model Context Info", lines=4)
            check_context_btn = gr.Button("Check Context Window Size")
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500
                # Removed incompatible parameters for Gradio 3.50.2
            )
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                container=False
            )
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
    
    # Set up event handlers
    check_context_btn.click(
        fn=check_context_size,
        inputs=model_dropdown,
        outputs=context_info
    )
    
    msg.submit(
        fn=chat_with_model,
        inputs=[msg, chatbot, model_dropdown],
        outputs=[chatbot]
    )
    
    submit_btn.click(
        fn=chat_with_model,
        inputs=[msg, chatbot, model_dropdown],
        outputs=[chatbot]
    )
    
    # Function to clear chat history
    def clear_chat():
        return []
    
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=chatbot
    )

if __name__ == "__main__":
    # Use the queue for more reliable async responses
    demo.queue().launch(share=True)
