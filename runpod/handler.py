import os
import sys
import asyncio
import traceback
import runpod
from typing import Dict, Any
from loguru import logger

from fish_speech.utils.schema import ServeMessage, ServeTextPart, ServeVQPart
from tools.fish_e2e import FishE2EAgent, FishE2EEventType

# Configure better logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/var/log/runpod_handler.log", rotation="100 MB", retention="1 week")

class ChatState:
    def __init__(self):
        self.conversation = []
        self.added_systext = False
        self.added_sysaudio = False

    def get_history(self):
        results = []
        for msg in self.conversation:
            results.append({"role": msg.role, "content": self.repr_message(msg)})
        return results

    def repr_message(self, msg: ServeMessage):
        response = ""
        for part in msg.parts:
            if isinstance(part, ServeTextPart):
                response += part.text
            elif isinstance(part, ServeVQPart):
                response += f"<audio {len(part.codes[0]) / 21:.2f}s>"
        return response


# Create a global agent once, instead of creating it for each request
agent = None

def get_agent():
    global agent
    if agent is None:
        try:
            import requests
            # Test if the API server is ready using synchronous request
            response = requests.get("http://localhost:8080/v1/health", timeout=5)
            if response.status_code != 200:
                logger.warning(f"API server health check failed: {response.status_code}")
            
            logger.info("Creating FishE2EAgent...")
            agent = FishE2EAgent()
            logger.info("FishE2EAgent created successfully")
        except Exception as e:
            logger.error(f"Error initializing FishE2EAgent: {str(e)}")
            # Continue without an agent - it might initialize correctly on the next attempt
    return agent


async def process_request(job_input):
    """Process a single request using FishE2EAgent"""
    try:
        agent = get_agent()
        if agent is None:
            raise RuntimeError("Failed to initialize FishE2EAgent")
            
        # Get conversation state
        conversation_id = job_input.get('conversation_id')
        state = ChatState()  # Create new state for each request
        
        # Get input parameters
        text_input = job_input.get('message')
        if not text_input:
            raise ValueError("No message provided")
            
        logger.info(f"Processing message: {text_input[:50]}...")
            
        sys_text = job_input.get('system_message')
        sys_audio_data = None  # Currently not handling system audio upload in the API
        
        # Process system message if provided
        if sys_text:
            logger.info(f"Adding system message: {sys_text[:50]}...")
            state.added_systext = True
            state.conversation.append(
                ServeMessage(
                    role="system",
                    parts=[ServeTextPart(text=sys_text)]
                )
            )
        
        # Process user input
        state.conversation.append(
            ServeMessage(
                role="user",
                parts=[ServeTextPart(text=text_input)]
            )
        )
        
        # Stream responses from agent
        text_response = ""
        audio_data = None
        
        logger.info("Streaming response from agent...")
        async for event in agent.stream(
            sys_audio_data=sys_audio_data,
            user_audio_data=None,  # Only handling text for now
            sample_rate=44100,
            num_channels=1,
            chat_ctx={
                "messages": state.conversation,
                "added_sysaudio": state.added_sysaudio,
            },
        ):
            if event.type == FishE2EEventType.TEXT_SEGMENT:
                text_response += event.text
            elif event.type == FishE2EEventType.SPEECH_SEGMENT:
                audio_data = event.frame.data
        
        # Store assistant response in conversation
        state.conversation.append(
            ServeMessage(
                role="assistant",
                parts=[ServeTextPart(text=text_response)]
            )
        )
            
        # Prepare result
        result = {
            "text": text_response,
            "history": state.get_history()
        }
        
        if audio_data is not None:
            result["audio"] = audio_data.tobytes()
                
        return {
            "status": "success",
            "output": result
        }

    except Exception as e:
        error_message = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error processing request: {error_message}\n{error_trace}")
        
        return {
            "status": "error",
            "output": {
                "error": error_message,
                "trace": error_trace
            }
        }

def handler(event):
    """RunPod handler function"""
    try:
        job_id = event.get('id', 'unknown')
        logger.info(f"Handling job {job_id}")
        
        input_data = event.get('input', {})
        if not input_data:
            return {
                "id": job_id,
                "status": "error",
                "output": {"error": "No input provided"}
            }
        
        # Run the async processing in the event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(process_request(input_data))
        
        return {
            "id": job_id,
            **result
        }
    except Exception as e:
        error_message = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error in handler: {error_message}\n{error_trace}")
        
        return {
            "id": event.get('id', 'unknown'),
            "status": "error",
            "output": {
                "error": error_message,
                "trace": error_trace
            }
        }

if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    try:
        # Wait for API server to be fully initialized
        logger.info("Waiting for API server to initialize...")
        import time
        time.sleep(5)  # Give the API server a bit more time to start
        
        # Create global agent instance to validate API server is ready
        get_agent()
        
        # Start RunPod handler
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.critical(f"Fatal error in runpod handler: {str(e)}")
        sys.exit(1)
