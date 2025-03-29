#!/usr/bin/env python
# RunPod Serverless handler for Fish-Speech Agent

import os
import sys
import json
import time
import asyncio
import base64
import traceback
import io
import wave
from typing import Dict, Any
from loguru import logger

# Add proper path for imports
sys.path.append("/app/fish-speech")

# Configure better logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/var/log/runpod_handler.log", rotation="100 MB", retention="1 week")

# Helper function for creating WAV headers - copied from e2e_webui.py
def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes

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
        sys_sr = 44100  # Default sample rate
        
        if job_input.get('system_audio'):
            try:
                import numpy as np
                import soundfile as sf
                
                # Decode base64 audio
                audio_bytes = base64.b64decode(job_input['system_audio'])
                with io.BytesIO(audio_bytes) as audio_buffer:
                    sys_audio_data, sys_sr = sf.read(audio_buffer)
                    # Convert to float32 if needed
                    if sys_audio_data.dtype != np.float32:
                        sys_audio_data = sys_audio_data.astype(np.float32)
                    
                    # Ensure audio is mono if it's stereo
                    if len(sys_audio_data.shape) > 1 and sys_audio_data.shape[1] > 1:
                        sys_audio_data = np.mean(sys_audio_data, axis=1)
                
                logger.info(f"Loaded system audio: {sys_audio_data.shape} at {sys_sr}Hz")
            except Exception as e:
                logger.error(f"Error processing system audio: {str(e)}")
                sys_audio_data = None
        
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
        
        # Stream responses from agent with retry logic
        text_response = ""
        audio_data = None
        
        # Initialize a fresh agent client for this request
        # This helps prevent stale connections
        try:
            # Close existing agent client if it exists
            if hasattr(agent, 'client'):
                await agent.client.aclose()
            
            # Create a new client with increased timeout
            import httpx
            agent.client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=60.0),  # Increased timeout
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0
                ),
            )
            logger.info("Created fresh HTTPX client for this request")
        except Exception as e:
            logger.warning(f"Could not refresh agent client: {str(e)}")
        
        logger.info("Streaming response from agent...")
        
        # Implement retry logic for stream
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Stream with the agent
                async for event in agent.stream(
                    sys_audio_data,     # System audio data for voice cloning
                    None,               # User audio data (None since we're using text)
                    sys_sr,             # Sample rate from the loaded audio or default
                    1,                  # Num channels (always 1 for RunPod API)
                    chat_ctx={
                        "messages": state.conversation,
                        "added_sysaudio": state.added_sysaudio,
                    },
                ):
                    if event.type == FishE2EEventType.TEXT_SEGMENT:
                        text_response += event.text
                        logger.debug(f"Text segment received: {event.text}")
                    elif event.type == FishE2EEventType.SPEECH_SEGMENT:
                        audio_data = event.frame.data
                        logger.debug(f"Speech segment received: {len(audio_data)} bytes")
                    elif event.type == FishE2EEventType.USER_CODES:
                        logger.debug(f"User VQ codes received")
                
                # If we got here, streaming completed successfully
                break
                
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout, 
                    httpx.ReadError, httpx.ConnectError) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to stream after {max_retries} attempts: {str(e)}")
                    if text_response:
                        # If we have a partial text response, let's continue with that
                        logger.warning(f"Using partial text response of {len(text_response)} chars")
                    else:
                        # If we don't even have text, raise the error
                        raise Exception(f"Network error when communicating with API server: {str(e)}")
                else:
                    logger.warning(f"Network error (attempt {retry_count}/{max_retries}): {str(e)}")
                    await asyncio.sleep(1)  # Brief pause before retrying
                    
                    # Recreate the client for the next attempt
                    try:
                        await agent.client.aclose()
                        agent.client = httpx.AsyncClient(
                            timeout=httpx.Timeout(300.0, connect=60.0),
                            limits=httpx.Limits(
                                max_keepalive_connections=5,
                                max_connections=10,
                                keepalive_expiry=30.0
                            ),
                        )
                    except Exception as client_err:
                        logger.warning(f"Could not refresh client: {str(client_err)}")
        
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
            
            # Create WAV header and append audio data for complete WAV file
            wav_header = wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1)
            complete_audio = wav_header + audio_data.tobytes()
            
            # Include base64 encoded WAV for direct use in web applications
            result["audio_base64"] = base64.b64encode(complete_audio).decode('utf-8')
            result["audio_format"] = "wav"
            result["sample_rate"] = 44100
            
            # Also include raw audio array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            result["audio"] = audio_array.tolist()
        
        logger.info(f"Response generated: {len(text_response)} chars of text, " + 
                    (f"{len(audio_data)} bytes of audio" if audio_data else "no audio"))
        
        # Make sure to close the client
        try:
            if hasattr(agent, 'client'):
                await agent.client.aclose()
        except Exception as e:
            logger.warning(f"Error closing client: {str(e)}")
            
        return {
            "status": "success", 
            "output": result
        }

    except Exception as e:
        error_message = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"Error processing request: {error_message}\n{error_trace}")
        
        # Clean up in case of error
        try:
            if 'agent' in locals() and hasattr(agent, 'client'):
                await agent.client.aclose()
        except Exception:
            pass
            
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
        
        # Fix for "This event loop is already running" error
        # Create a new event loop for each request instead of using the default one
        try:
            # Create a new event loop for this handler
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result = asyncio.run(process_request(input_data))
        except RuntimeError:
            # If we still have an issue with the event loop, try a different approach
            logger.warning("Event loop error detected, using alternative approach")
            # Use nest_asyncio as a fallback if available
            try:
                import nest_asyncio
                nest_asyncio.apply()
                result = asyncio.get_event_loop().run_until_complete(process_request(input_data))
            except ImportError:
                # If nest_asyncio is not available, use a thread-based approach
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(process_request(input_data)))
                    result = future.result()
        
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
    # Try to install nest_asyncio if not available (helps with event loop issues)
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        logger.info("Installing nest_asyncio to help with event loop issues...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("nest_asyncio installed and applied successfully")
        except Exception as e:
            logger.warning(f"Could not install or apply nest_asyncio: {str(e)}")
    
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
