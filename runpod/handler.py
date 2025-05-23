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
import re  # Added for question extraction
import httpx
import tempfile
from typing import Dict, Any, AsyncGenerator, Union
from loguru import logger

# Import the missing rtc module
try:
    import livekit.rtc as rtc
except ImportError:
    # Attempt to install livekit.rtc if missing
    try:
        import subprocess
        logger.info("Installing livekit-rtc module...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "livekit"])
        import livekit.rtc as rtc
    except Exception as e:
        logger.error(f"Failed to install livekit-rtc: {e}")
        # Provide a fallback implementation if necessary
        class MockRTC:
            class AudioFrame:
                def __init__(self, data, samples_per_channel, sample_rate, num_channels):
                    self.data = data
                    self.samples_per_channel = samples_per_channel
                    self.sample_rate = sample_rate
                    self.num_channels = num_channels
        rtc = MockRTC()

# Add proper path for imports
sys.path.append("/app/fish-speech")

# Configure better logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/var/log/runpod_handler.log", rotation="100 MB", retention="1 week")

# Load configuration from environment
MODEL_PATH = os.getenv("MODEL_PATH", "/app/fish-speech/checkpoints")
LLAMA_CHECKPOINT = os.getenv("LLAMA_CHECKPOINT_PATH", "/app/fish-speech/checkpoints/fish-speech-1.5/")
DECODER_CHECKPOINT = os.getenv("DECODER_CHECKPOINT_PATH", "/app/fish-speech/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
DECODER_CONFIG = os.getenv("DECODER_CONFIG_NAME", "base")
USE_HALF = os.getenv("HALF", "true").lower() == "true"
USE_COMPILE = os.getenv("COMPILE", "true").lower() == "true"
ASR_ENABLED = os.getenv("ASR_ENABLED", "false").lower() == "true"
# Check server mode from environment variable
SERVER_MODE = os.getenv("SERVER_MODE", "tts")
logger.info(f"Server mode: {SERVER_MODE}")

# Helper functions for audio handling
def create_audio_frame(data: bytes, samples_per_channel: int) -> rtc.AudioFrame:
    """Create a LiveKit AudioFrame from raw audio data."""
    return rtc.AudioFrame(
        data=data,
        samples_per_channel=samples_per_channel,
        sample_rate=44100,  # Fixed for Fish-Speech
        num_channels=1
    )

def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    """Create a WAV header for the given audio parameters."""
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
    from tools.schema import ServeMessage, ServeTextPart, ServeVQPart, ServeTTSRequest, ServeReferenceAudio
    from tools.fish_e2e import FishE2EAgent, FishE2EEventType
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    logger.error("Attempting to install missing packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod==1.7.7"])
    import runpod
    from tools.schema import ServeMessage, ServeTextPart, ServeVQPart, ServeTTSRequest, ServeReferenceAudio
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

        # Process assistant messages to extract questions and update user messages
        for i, msg in enumerate(results):
            if msg["role"] == "assistant":
                match = re.search(r"Question: (.*?)\n\nResponse:", msg["content"])
                if match and i > 0 and results[i - 1]["role"] == "user":
                    # Update previous user message with extracted question
                    results[i - 1]["content"] += "\n" + match.group(1)
                    # Remove the Question/Answer format from assistant message
                    msg["content"] = msg["content"].split("\n\nResponse: ", 1)[1]
        return results

    def repr_message(self, msg: ServeMessage):
        response = ""
        for part in msg.parts:
            if isinstance(part, ServeTextPart):
                response += part.text
            elif isinstance(part, ServeVQPart):
                response += f"<audio {len(part.codes[0]) / 21:.2f}s>"
        return response
        
    def append_message(self, part: ServeTextPart | ServeVQPart, role: str = "assistant") -> None:
        """Helper method to append messages in a consistent way matching e2e_webui.py"""
        if not self.conversation or self.conversation[-1].role != role:
            self.conversation.append(ServeMessage(role=role, parts=[part]))
        else:
            self.conversation[-1].parts.append(part)


# Create a global agent instance to avoid reloading the model for each request
_AGENT = None

def get_agent(max_retries: int = 30, retry_delay: int = 2):
    """Initialize the agent if it doesn't exist"""
    global _AGENT
    if _AGENT is None:
        try:
            # Wait for API server to be ready
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Try POST request instead of GET since the endpoint returns 405 for GET
                    import urllib.request
                    req = urllib.request.Request(
                        "http://localhost:8080/v1/health",
                        method="POST",  # Change from GET to POST
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=5) as response:
                        if response.status == 200:
                            logger.info("API server is ready")
                            break
                        logger.warning(f"API server health check failed: {response.status}")
                except urllib.error.HTTPError as e:
                    # Handle 405 Method Not Allowed specifically
                    if e.code == 405:
                        logger.info("API server is ready (responded with 405, which means endpoint exists)")
                        break
                    logger.warning(f"Health check HTTP error: {e.code} - {e.reason}")
                except Exception as e:
                    logger.warning(f"Health check attempt {retry_count + 1}/{max_retries} failed: {str(e)}")
                
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"Timeout waiting for API server after {max_retries} attempts")
                time.sleep(retry_delay)

            # Initialize the agent with environment configuration
            logger.info("Creating FishE2EAgent...")
            agent = FishE2EAgent()
            
            # Configure agent based on environment
            agent.llama_url = "http://localhost:8080/v1/chat"
            agent.vqgan_url = "http://localhost:8080"
            
            # Initialize HTTPX client with proper configuration
            agent.client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=60.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0
                ),
            )

            # Verify model paths from environment
            if not os.path.exists(LLAMA_CHECKPOINT):
                raise RuntimeError(f"LLAMA checkpoint not found at {LLAMA_CHECKPOINT}")
            if not os.path.exists(DECODER_CHECKPOINT):
                raise RuntimeError(f"Decoder checkpoint not found at {DECODER_CHECKPOINT}")
            
            _AGENT = agent
            logger.info("FishE2EAgent created and configured successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FishE2EAgent: {str(e)}")
            raise
            
    return _AGENT


async def process_request(job_input: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """Process a text-to-speech request using either direct TTS endpoint or FishE2EAgent.
    
    Args:
        job_input: Dictionary containing:
            - message (str): Text to convert to speech
            - system_message (str, optional): System prompt to control behavior
            - system_audio (str, optional): Base64 encoded reference audio for voice cloning
            - conversation_id (str, optional): ID to maintain context
            - streaming (bool, optional): Enable streaming response (default: True)
            - format (str, optional): Audio format (wav/mp3/flac, default: wav)
            - temperature (float, optional): Sampling temperature (default: 0.7)
            - max_new_tokens (int, optional): Max tokens to generate (default: 512)
            - max_text_length (int, optional): Max input text length (default: 1000)
            - tts (bool, optional): Force using TTS endpoint regardless of server mode
            
    Yields:
        Dict[str, Any]: Stream of response chunks containing:
            type: "text" or "audio"
            content: Text segment or base64-encoded audio
            
    Raises:
        ValueError: For validation errors
        RuntimeError: For server/model errors
        Exception: For unexpected errors
    """
    try:
        # Check if we should use direct TTS mode either due to server mode or explicit request
        use_tts_endpoint = SERVER_MODE == "tts" or job_input.get('tts', False)
        
        # Get input parameters
        text_input = job_input.get('message')
        if not text_input:
            raise ValueError("No message provided in the input")
            
        # Process system audio if provided (base64 encoded)
        system_audio_b64 = job_input.get('system_audio')
        
        # If using direct TTS mode
        if use_tts_endpoint:
            logger.info("Using direct TTS endpoint for voice cloning...")
            
            if not system_audio_b64:
                raise ValueError("No system_audio provided for TTS cloning")
                
            # Prepare HTTP client for TTS request
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=60.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            try:
                # Decode base64 to get raw bytes
                audio_bytes = base64.b64decode(system_audio_b64)
                logger.info(f"[DEBUG] Audio data before sending to TTS: {len(audio_bytes)} bytes")
                
                # Try to extract audio format and other properties for debugging
                with io.BytesIO(audio_bytes) as audio_buffer:
                    try:
                        import wave
                        with wave.open(audio_buffer, 'rb') as wav_file:
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                            frame_rate = wav_file.getframerate()
                            frames = wav_file.getnframes()
                            duration = frames / frame_rate
                            logger.info(f"[DEBUG] WAV properties: channels={channels}, sample_width={sample_width}, "
                                        f"frame_rate={frame_rate}, frames={frames}, duration={duration:.2f}s")
                    except Exception as wav_err:
                        logger.warning(f"[DEBUG] Not a valid WAV file or error reading WAV: {str(wav_err)}")

                # Prepare the payload according to ServeTTSRequest schema - use raw bytes, not base64
                # The key fix: Using the text input as the reference text for proper voice cloning
                reference_text = job_input.get('reference_text', text_input)  # Use provided reference text or input text
                
                # Import the ServeReferenceAudio from schema.py if not already imported
                try:
                    from tools.schema import ServeReferenceAudio, ServeTTSRequest
                except ImportError:
                    # If schema isn't imported yet, try importing it 
                    sys.path.append("/app/fish-speech")
                    from tools.schema import ServeReferenceAudio, ServeTTSRequest
                
                # Create a proper ServeReferenceAudio object
                ref_audio = ServeReferenceAudio(audio=audio_bytes, text=reference_text)

                # --- Construct payload using ServeTTSRequest model ---
                # Get optional parameters from job_input with defaults matching post_api.py or common usage
                tts_data = {
                    "text": text_input,
                    "references": [ref_audio],
                    "reference_id": job_input.get('reference_id', None), # Add missing field
                    "normalize": job_input.get('normalize', True),
                    "format": job_input.get('format', 'wav'),
                    "mp3_bitrate": job_input.get('mp3_bitrate', 64), # Add missing field
                    "opus_bitrate": job_input.get('opus_bitrate', -1000), # Add missing field
                    "max_new_tokens": job_input.get('max_new_tokens', 0), # Default 0 like post_api
                    "chunk_length": job_input.get('chunk_length', 200), # Use job_input value
                    "top_p": job_input.get('top_p', 0.7), # Default 0.7 like post_api
                    "repetition_penalty": job_input.get('repetition_penalty', 1.2),
                    "temperature": job_input.get('temperature', 0.7), # Default 0.7 like post_api
                    "streaming": job_input.get('streaming', False), # Use job_input value, default False
                    "use_memory_cache": job_input.get('use_memory_cache', 'never'), # Use job_input value, default 'never' like post_api
                    "seed": job_input.get('seed', None), # Add missing field
                }

                # Create the Pydantic model instance
                try:
                    tts_request_model = ServeTTSRequest(**tts_data)
                    logger.info(f"Constructed ServeTTSRequest model: {tts_request_model}")
                except Exception as pydantic_err:
                     logger.error(f"Pydantic validation error creating ServeTTSRequest: {pydantic_err}")
                     raise ValueError(f"Input validation failed for TTS request: {pydantic_err}")

                logger.info(f"Sending TTS request to endpoint for text: {text_input[:50]}...")

                # Use MessagePack serialization instead of JSON
                import ormsgpack

                # Make the request to the TTS endpoint with MessagePack, serializing the Pydantic model
                response = await client.post(
                    "http://localhost:8080/v1/tts",
                    content=ormsgpack.packb(tts_request_model, option=ormsgpack.OPT_SERIALIZE_PYDANTIC), # Serialize the model instance
                    headers={"Content-Type": "application/msgpack"},
                    timeout=300.0
                )
                
                logger.info(f"Received response from TTS endpoint: {response}")

                # Handle unsuccessful responses
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_message = error_data.get('message', 'Unknown error')
                    except:
                        error_message = response.text
                    logger.error(f"TTS endpoint error ({response.status_code}): {error_message}")
                    raise RuntimeError(f"TTS endpoint failed with status {response.status_code}: {error_message}")
                    
                # Get the audio data from the response
                audio_data = await response.aread()
                
                if not audio_data or len(audio_data) < 100:  # Check for very small responses that are likely errors
                    raise RuntimeError(f"TTS endpoint returned empty or invalid response (size: {len(audio_data) if audio_data else 0} bytes)")
                    
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                logger.info(f"Successfully received {len(audio_data)} bytes of audio from TTS endpoint")
                
                # Return the complete result using the "summary" type expected by the handler's non-streaming collector
                yield {
                    "type": "summary", # Add the type key
                    "content": {       # Put the actual output data inside "content"
                        "text": text_input,
                        "audio_base64": audio_base64, # Keep base64 key for now, handler will rename
                        "audio_format": job_input.get('format', 'wav'),
                        "sample_rate": 44100,
                        "status": "success" # Explicitly add success status here
                    }
                }

                # Clean up client
                await client.aclose()
                return # Return after yielding the single summary chunk
                
            except Exception as e:
                # Clean up client in case of error
                await client.aclose()
                logger.error(f"Error using TTS endpoint: {str(e)}")
                raise RuntimeError(f"Error using TTS endpoint: {str(e)}")
        
        # If not using TTS endpoint, use the agent as before (original code follows)
        logger.info("Using FishE2EAgent for voice interaction...")
        
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
        
        max_text_length = job_input.get('max_text_length', 1000)
        if len(text_input) > max_text_length:
            raise ValueError(f"Text is too long, max length is {max_text_length}")

        logger.info(f"Processing message: {text_input[:50]}...")
        
        # Handle optional parameters
        sys_text = job_input.get('system_message')
        conversation_id = job_input.get('conversation_id')
        streaming = job_input.get('streaming', True)
        output_format = job_input.get('format', 'wav')
        
        if streaming and output_format != 'wav':
            raise ValueError("Streaming only supports WAV format")

        # Process system audio if provided (base64 encoded)
        sys_audio_data = None
        sys_sr = 44100  # Default sample rate for Fish-Speech
        
        # Process system audio if provided (base64 encoded)
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
                
                # Validate audio length
                max_audio_duration = 90  # Maximum 90 seconds as per docs
                if len(sys_audio_data) / sys_sr > max_audio_duration:
                    raise ValueError(f"System audio is too long (max {max_audio_duration}s)")

                logger.info(f"Loaded system audio: {sys_audio_data.shape} at {sys_sr}Hz")
                state.added_sysaudio = True
            except Exception as e:
                logger.error(f"Error processing system audio: {str(e)}")
                sys_audio_data = None
        
        # Process system message if provided
        if sys_text:
            logger.info(f"Adding system message: {sys_text[:50]}...")
            state.added_systext = True
            state.append_message(ServeTextPart(text=sys_text), role="system")
        else:
            # Add default system message for Fish-Speech
            default_system_message = os.getenv(
                "DEFAULT_SYSTEM_MESSAGE",
                'You are a voice assistant designed by Fish Audio, providing end-to-end voice interaction for seamless user experience.'
            )
            state.append_message(
                ServeTextPart(text=default_system_message),
                role="system"
            )
        
        # Add user message to conversation
        state.append_message(ServeTextPart(text=text_input), role="user")

        # Configure generation parameters
        gen_config = {
            "max_new_tokens": job_input.get('max_new_tokens', 512),
            "temperature": job_input.get('temperature', 0.7),
            "top_p": job_input.get('top_p', 0.9),
            "repetition_penalty": job_input.get('repetition_penalty', 1.2),
            "early_stop_threshold": job_input.get('early_stop_threshold', 0.5),
            "chunk_length": job_input.get('chunk_length', 200),
        }
        
        # Configure HTTPX client with proper timeout and retry settings
        try:
            if hasattr(agent, 'client'):
                await agent.client.aclose()
            
            agent.client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=60.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0
                ),
            )
            logger.info("Created fresh HTTPX client for this request")
        except Exception as e:
            logger.warning(f"Could not refresh agent client: {str(e)}")

        # Initialize response containers
        text_segments = []
        audio_segments = []
        vq_codes_history = []
        current_text = ""
        result_audio = b""
        
        logger.info(f"Streaming response from agent...")
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Stream with the agent
                async for event in agent.stream(
                    sys_audio_data,     # System audio for voice cloning
                    None,               # No user audio since using text
                    sys_sr,            # Sample rate 
                    1,                 # Mono audio
                    chat_ctx={
                        "messages": state.conversation,
                        "added_sysaudio": state.added_sysaudio,
                    },
                ):
                    if event.type == FishE2EEventType.TEXT_SEGMENT:
                        text_segments.append(event.text)
                        current_text += event.text
                        state.append_message(ServeTextPart(text=event.text))
                        
                        if streaming:
                            yield {
                                "type": "text",
                                "content": event.text
                            }
                            
                    elif event.type == FishE2EEventType.SPEECH_SEGMENT:
                        # In agent-inference branch, this is already a LiveKit rtc.AudioFrame
                        audio_frame = event.frame
                        audio_segments.append(audio_frame)
                        result_audio += audio_frame.data
                        
                        # Store vq_codes from the event if available
                        if hasattr(event, 'vq_codes') and event.vq_codes:
                            vq_codes_history.append(event.vq_codes)
                            # Add VQ codes to conversation state
                            state.append_message(ServeVQPart(codes=event.vq_codes))
                        
                        if streaming and output_format == 'wav':
                            # Create WAV chunk
                            wav_chunk = wav_chunk_header(
                                sample_rate=44100,
                                bit_depth=16,
                                channels=1
                            ) + audio_frame.data
                            
                            yield {
                                "type": "audio",
                                "content": base64.b64encode(wav_chunk).decode('utf-8')
                            }
                            
                    elif event.type == FishE2EEventType.USER_CODES:
                        if event.vq_codes:
                            state.append_message(ServeVQPart(codes=event.vq_codes), role="user")
                
                # Successfully processed stream
                break
                
            except (httpx.RemoteProtocolError, httpx.ReadTimeout,
                    httpx.ConnectTimeout, httpx.ReadError,
                    httpx.ConnectError) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to stream after {max_retries} attempts: {str(e)}")
                    if text_segments:  # Return partial response if we have any
                        break
                    raise Exception(f"Network error when communicating with API server: {str(e)}")
                    
                logger.warning(f"Network error (attempt {retry_count}/{max_retries}): {str(e)}")
                await asyncio.sleep(1)
                
                # Refresh client connection
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

        # Prepare final result
        if audio_segments:
            import numpy as np
            
            # Create complete WAV file using frame data
            wav_header = wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1)
            complete_audio = wav_header + result_audio
            
            # Generate a final summary result chunk
            final_result = {
                "type": "summary", 
                "content": {
                    "text": current_text,
                    "audio_base64": base64.b64encode(complete_audio).decode('utf-8'),
                    "audio_format": output_format,
                    "sample_rate": 44100,
                    "history": state.get_history()
                }
            }
            
            # Also include raw audio array for further processing
            audio_array = np.frombuffer(result_audio, dtype=np.int16)
            
            # Calculate duration for logging
            audio_duration = len(audio_array) / 44100
            logger.info(
                f"Generated {len(current_text)} chars of text and "
                f"{len(result_audio)} bytes of audio "
                f"(duration: {audio_duration:.2f}s)"
            )
            
            # Include vq_codes if available for advanced clients
            if vq_codes_history:
                final_result["content"]["vq_codes"] = vq_codes_history
                
            # Yield the final summary instead of returning
            yield final_result
        else:
            logger.warning("No audio generated in response")
            # Yield a warning message
            yield {
                "type": "warning",
                "content": "No audio could be generated for this response"
            }

        # Clean up client connection
        try:
            if hasattr(agent, 'client'):
                await agent.client.aclose()
        except Exception as e:
            logger.warning(f"Error closing client: {str(e)}")

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
            
        # Yield error information instead of returning
        yield {
            "type": "error",
            "content": {
                "error": error_message,
                "trace": error_trace
            }
        }


def get_error_response(job_id: str, error: Exception, status_code: int = 500) -> dict:
    """Generate a standardized error response"""
    error_message = str(error)
    error_type = type(error).__name__
    error_trace = traceback.format_exc() if status_code >= 500 else None
    
    logger.error(f"{error_type} processing request: {error_message}")
    if error_trace:
        logger.error(error_trace)
    
    response = {
        "id": job_id,
        "status": "error",
        "output": {
            "error": error_message,
            "error_type": error_type
        }
    }
    
    if error_trace:
        response["output"]["trace"] = error_trace
        
    return response

def handler(event):
    """
    RunPod handler function - processes requests through the Fish-Speech agent
    """
    job_id = event.get('id', 'unknown')
    logger.info(f"Handling job {job_id}")
    
    # Input validation
    input_data = event.get('input', {})
    if not input_data:
        return get_error_response(
            job_id,
            ValueError("No input provided"),
            status_code=400
        )
    
    try:
        # Check if streaming is requested
        streaming = input_data.get('streaming', True)
        
        # If streaming is enabled, use RunPod's generator response
        if streaming and not os.environ.get("RUNPOD_LOCAL_TEST") == "1":
            # Collect stream chunks into the output format
            async def collect_stream():
                chunks = []
                audio_data = None
                text_content = ""
                
                try:
                    async for chunk in process_request(input_data):
                        if chunk["type"] == "text":
                            text_content += chunk["content"]
                        elif chunk["type"] == "audio":
                            audio_data = chunk["content"]  # Keep the latest audio chunk
                        elif chunk["type"] == "summary":
                            # Return final summary directly
                            return {"output": chunk["content"]}
                        chunks.append(chunk)
                except Exception as e:
                    return {"output": {
                        "error": str(e),
                        "status": "error"
                    }}

                # Format final output
                return {
                    "output": {
                        "text": text_content,
                        "audio": audio_data,
                        "chunks": chunks,
                        "status": "success"
                    }
                }

            # Return results in RunPod format
            loop = asyncio.get_event_loop()
            final_result = loop.run_until_complete(collect_stream())
            return final_result
        
        # For non-streaming requests, collect all outputs
        else:
            # Format output data according to RunPod requirements
            result_data = {
                "output": {
                    "text": "",
                    "audio": None,
                    "audio_format": input_data.get('format', 'wav'),
                    "sample_rate": 44100,
                    "status": "success"
                }
            }
            
            # Handle event loop configuration
            if os.environ.get("RUNPOD_USE_NEST_ASYNCIO", "0") == "1":
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    
                    # Collect all chunks from the generator
                    async def collect_nonstreaming():
                        async for chunk in process_request(input_data):
                            if chunk["type"] == "text":
                                result_data["output"]["text"] += chunk["content"]
                            elif chunk["type"] == "audio" and not result_data["output"]["audio"]:
                                result_data["output"]["audio"] = chunk["content"]
                            elif chunk["type"] == "summary":
                                # Use summary content if available
                                summary_content = chunk["content"]
                                # Check if audio_base64 exists and rename it to 'audio' for the final output structure
                                if "audio_base64" in summary_content:
                                     summary_content["audio"] = summary_content.pop("audio_base64")
                                # Ensure status is present
                                if "status" not in summary_content:
                                     summary_content["status"] = "success"
                                return {"output": summary_content} # Return the expected structure
                            elif chunk["type"] == "error": # Handle potential errors yielded
                                 return {"output": chunk["content"], "status": "error"}
                        # If loop finishes without a summary or error, return the collected data
                        # This path might be taken if the TTS yield isn't caught correctly
                        return result_data
                    
                    result = asyncio.get_event_loop().run_until_complete(collect_nonstreaming())
                except ImportError:
                    logger.warning("nest_asyncio not available, falling back to new event loop")
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    
                    async def collect_nonstreaming():
                        async for chunk in process_request(input_data):
                            if chunk["type"] == "text":
                                result_data["output"]["text"] += chunk["content"]
                            elif chunk["type"] == "audio" and not result_data["output"]["audio"]:
                                result_data["output"]["audio"] = chunk["content"]
                            elif chunk["type"] == "summary":
                                return {"output": chunk["content"]}
                        return result_data
                    
                    result = asyncio.run(collect_nonstreaming())
            else:
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    
                    async def collect_nonstreaming():
                        async for chunk in process_request(input_data):
                            if chunk["type"] == "text":
                                result_data["output"]["text"] += chunk["content"]
                            elif chunk["type"] == "audio" and not result_data["output"]["audio"]:
                                result_data["output"]["audio"] = chunk["content"]
                            elif chunk["type"] == "summary":
                                return {"output": chunk["content"]}
                        return result_data
                    
                    result = asyncio.run(collect_nonstreaming())
                except RuntimeError as e:
                    if "This event loop is already running" in str(e):
                        logger.warning("Event loop already running, using thread-based approach")
                        import concurrent.futures
                        
                        def run_async_collection():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            async def collect_nonstreaming():
                                nonlocal result_data
                                async for chunk in process_request(input_data):
                                    if chunk["type"] == "text":
                                        result_data["output"]["text"] += chunk["content"]
                                    elif chunk["type"] == "audio" and not result_data["output"]["audio"]:
                                        result_data["output"]["audio"] = chunk["content"]
                                    elif chunk["type"] == "summary":
                                        return {"output": chunk["content"]}
                                return result_data
                            
                            return loop.run_until_complete(collect_nonstreaming())
                        
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_async_collection)
                            result = future.result()
                    else:
                        raise

            return {
                "id": job_id,
                **result
            }

    except ValueError as e:
        # Handle validation errors
        return get_error_response(job_id, e, status_code=400)
        
    except (httpx.NetworkError, httpx.TimeoutException) as e:
        # Handle network/timeout errors
        return get_error_response(job_id, e, status_code=503)
        
    except Exception as e:
        # Handle all other errors
        return get_error_response(job_id, e)


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
        runpod.serverless.start({
            "handler": handler,
            "return_aggregate_stream": True,
        })
