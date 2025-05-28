#!/usr/bin/env python
"""
API Server Integration for Ollama Gradio WebUI
---------------------------------------------
This script provides API endpoint access to the Ollama Gradio WebUI
using ngrok for public exposure from environments like Kaggle.
"""

import os
import sys
import gradio as gr
import argparse
import logging
import json
from pyngrok import ngrok, conf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_ngrok(port=7860, authtoken=None, region="us"):
    """Setup ngrok tunnel to expose the Gradio API"""
    # Configure ngrok (optional auth token)
    if authtoken:
        conf.get_default().auth_token = authtoken
    
    # Set up the ngrok connection
    try:
        # Close any existing tunnels
        for tunnel in ngrok.get_tunnels():
            ngrok.disconnect(tunnel.public_url)
        
        # Set up a new tunnel
        listener = ngrok.connect(
            addr=f"localhost:{port}",
            region=region
        )
        
        # Display the public URL
        public_url = listener.public_url
        logger.info(f"✅ API Server running at: {public_url}")
        logger.info(f"✅ Direct API endpoints:")
        logger.info(f"   - Chat API: {public_url}/api/chat")
        logger.info(f"   - React Agent API: {public_url}/api/react_agent")
        
        return public_url
    except Exception as e:
        logger.error(f"❌ Failed to establish ngrok tunnel: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start API server with ngrok tunnel")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--authtoken", type=str, default=None, help="ngrok authtoken (optional)")
    parser.add_argument("--region", type=str, default="us", help="ngrok region")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle mode")
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = str(args.port)
    
    if args.kaggle:
        os.environ["KAGGLE_API_MODE"] = "true"
        os.environ["OLLAMA_USE_VLLM"] = "true"
    
    # Set up ngrok tunnel
    public_url = setup_ngrok(port=args.port, authtoken=args.authtoken, region=args.region)
    if not public_url:
        logger.error("Failed to establish ngrok tunnel. Exiting.")
        sys.exit(1)
    
    # Start the vLLM app - import here to avoid circular imports
    try:
        from vllm_app import app
        logger.info("Starting Gradio application with API endpoints...")
        app.launch(server_name="0.0.0.0", server_port=args.port, share=False)
    except ImportError:
        logger.error("Failed to import vllm_app. Make sure you're in the correct directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
