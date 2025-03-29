#!/usr/bin/env python
# RunPod Serverless handler for Fish-Speech Agent

import os
import sys
import json
import time
import asyncio
import base64
import traceback
from typing import Dict, Any
from loguru import logger

# Add proper path for imports
sys.path.append("/app/fish-speech")

# Configure better logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/var/log/runpod_handler.log", rotation="100 MB", retention="1 week")

try:
    import runpod
    from fish_speech.utils.schema import ServeMessage, ServeTextPart, ServeVQPart
    from tools.fish_e2e import FishE2EAgent, FishE2EEventType
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    logger.error("Attempting to install missing packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod==1.7.0"])
    import runpod
    from fish_speech.utils.schema import ServeMessage, ServeTextPart, ServeVQPart
    from tools.fish_e2e import FishE2EAgent, FishE2EEventType


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


# Create a global agent instance to avoid reloading the model for each request
_AGENT = None

def get_agent():
    """Initialize the agent if it doesn't exist"""
    global _AGENT
    if _AGENT is None:
        try:
            # Health check using standard library
            import urllib.request
            try:
                with urllib.request.urlopen("http://localhost:8080/v1/health", timeout=5) as response:
                    if response.status != 200:
                        logger.warning(f"API server health check failed: {response.status}")
            except Exception as e:
                logger.warning(f"Health check failed: {str(e)}")
                
            # Initialize the agent
            logger.info("Creating FishE2EAgent...")
            _AGENT = FishE2EAgent()
            logger.info("FishE2EAgent created successfully")
        except Exception as e:
            logger.error(f"Error initializing FishE2EAgent: {str(e)}")
            
    return _AGENT


async def process_request(job_input):
    """Process a request using FishE2EAgent"""
    try:
        # Get the agent
        agent = get_agent()
        if agent is None:
            raise RuntimeError("Failed to initialize FishE2EAgent")
        
        # Create state for this conversation
        state = ChatState()
        
        # Get input parameters
        text_input = job_input.get('message')
        if not text_input:
            raise ValueError("No message provided in the input")
        
        logger.info(f"Processing message: {text_input[:50]}...")
        
        # Handle optional parameters
        sys_text = job_input.get('system_message')
        conversation_id = job_input.get('conversation_id')
        
        # Process system audio if provided (base64 encoded)
        sys_audio_data = None
        if job_input.get('system_audio'):
            try:
                import numpy as np
                import io
                import soundfile as sf
                
                # Decode base64 audio
                audio_bytes = base64.b64decode(job_input['system_audio'])
                with io.BytesIO(audio_bytes) as audio_buffer:
                    sys_audio_data, sample_rate = sf.read(audio_buffer)
                    # Convert to float32 if needed
                    if sys_audio_data.dtype != np.float32:
                        sys_audio_data = sys_audio_data.astype(np.float32)
                    
                    # Ensure audio is mono if it's stereo
                    if len(sys_audio_data.shape) > 1 and sys_audio_data.shape[1] > 1:
                        sys_audio_data = np.mean(sys_audio_data, axis=1)
                
                logger.info(f"Loaded system audio: {sys_audio_data.shape}")
            except Exception as e:
                logger.error(f"Error processing system audio: {str(e)}")
        
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
        
        # Add user message to conversation
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
            user_audio_data=None,
            sample_rate=44100,
            num_channels=1,
            chat_ctx={
                "messages": state.conversation,
                "added_sysaudio": state.added_sysaudio,
            },
        ):
            if event.type == FishE2EEventType.TEXT_SEGMENT:
                text_response += event.text
                logger.debug(f"Text segment: {event.text}")
            elif event.type == FishE2EEventType.SPEECH_SEGMENT:
                audio_data = event.frame.data
                logger.debug(f"Speech segment received: {len(audio_data)} bytes")
        
        # Store assistant response in conversation
        state.conversation.append(
            ServeMessage(
                role="assistant",
                parts=[ServeTextPart(text=text_response)]
            )
        )
        
        # Prepare the result
        result = {
            "text": text_response,
            "history": state.get_history()
        }
        
        # Include audio if available, converting to bytes for JSON serialization
        if audio_data is not None:
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            result["audio"] = audio_array.tolist()
            # Also provide base64 encoded audio for direct use in web applications
            audio_bytes = audio_data.tobytes()
            result["audio_base64"] = base64.b64encode(audio_bytes).decode('utf-8')
            result["audio_format"] = "wav"
            result["sample_rate"] = 44100
        
        logger.info(f"Response generated: {len(text_response)} chars of text, " + 
                    (f"{len(audio_data)} bytes of audio" if audio_data else "no audio"))
        
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
    """
    RunPod handler function - processes requests through the Fish-Speech agent
    """
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
        
        # Run the async processing in an event loop
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


# This allows local testing of the handler
def local_test():
    """Run a local test of the handler"""
    logger.info("Running local test")
    
    # Check if test_input.json exists, otherwise use default input
    test_input_path = os.path.join(os.path.dirname(__file__), "test_input.json")
    if os.path.exists(test_input_path):
        with open(test_input_path, "r") as f:
            test_event = json.load(f)
    else:
        test_event = {
            "id": "local_test",
            "input": {
                "message": "Hello, how are you today?",
                "system_message": "You are a helpful assistant."
            }
        }
    
    # Process the test event
    result = handler(test_event)
    
    # Print the result
    logger.info(f"Test result: {json.dumps(result, indent=2)}")
    
    # Save the output if it includes audio
    if result.get("status") == "success" and result.get("output", {}).get("audio_base64"):
        output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the text response
        with open(os.path.join(output_dir, "response.txt"), "w") as f:
            f.write(result["output"]["text"])
        
        # Save the audio if present
        if result["output"].get("audio_base64"):
            audio_bytes = base64.b64decode(result["output"]["audio_base64"])
            with open(os.path.join(output_dir, "response.wav"), "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Audio response saved to {os.path.join(output_dir, 'response.wav')}")


if __name__ == "__main__":
    # Add a delay to ensure the API server is ready
    logger.info("Starting RunPod serverless handler...")
    logger.info("Waiting for API server to initialize...")
    time.sleep(5)
    
    # Check if we're running a local test or as a serverless handler
    if os.environ.get("RUNPOD_LOCAL_TEST") == "1":
        local_test()
    else:
        # Start the RunPod serverless handler
        logger.info("Starting RunPod serverless handler")
        runpod.serverless.start({"handler": handler})
