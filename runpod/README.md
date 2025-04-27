# Fish Speech RunPod API

This guide explains how to use the Fish Speech API endpoint (ID: g7oisc3won8ng8) in your applications.
Runpod Serverless Guide : https://docs.runpod.io/serverless/endpoints/job-operations

## Technical Overview

Fish Speech processes requests through several steps:
1. VQGAN encoder for reference voice encoding
2. Language model for text-to-semantic token generation
3. VQGAN decoder for speech synthesis
4. Stream management for real-time responses

## Operation Modes

Fish Speech supports two operation modes:

### 1. Agent Mode (Default)
In this mode, the service functions as a conversational AI that can conduct multi-turn dialogues with voice responses matching your reference audio.

### 2. TTS Mode (Direct Voice Cloning)
This mode provides direct Text-to-Speech synthesis with your reference voice, skipping the conversational aspects. This is ideal for pure voice cloning applications where you simply want text spoken in a specific voice.

## Basic Usage

```python
import requests
import base64

ENDPOINT = "https://api.runpod.ai/v2/g7oisc3won8ng8/run"
API_KEY = "YOUR_API_KEY"

# Simple text-to-speech request
response = requests.post(
    ENDPOINT,
    json={
        "input": {
            "message": "Hello, how are you?",  # Use 'message' not 'text'
            "format": "wav",
            "streaming": False
        }
    },
    headers={"Authorization": f"Bearer {API_KEY}"}
)
```

## Voice Cloning

### Agent Mode Voice Cloning
```python
def clone_voice_agent(text, reference_audio_path):
    # Read and encode reference audio
    with open(reference_audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    response = requests.post(
        ENDPOINT,
        json={
            "input": {
                "message": text,
                "system_audio": audio_base64,  # Voice reference
                "format": "wav"
            }
        },
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    return response.json()
```

### Direct TTS Voice Cloning (Recommended for Pure Voice Cloning)
```python
def clone_voice_tts(text, reference_audio_path, reference_text):
    """
    Clone a voice using direct TTS synthesis.
    
    Args:
        text: The text to be spoken in the cloned voice
        reference_audio_path: Path to the reference audio file
        reference_text: Text that matches the reference audio content (required)
    """
    # Read and encode reference audio
    with open(reference_audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    if not reference_text:
        raise ValueError("reference_text is required for TTS voice cloning")
    
    response = requests.post(
        ENDPOINT,
        json={
            "input": {
                "text": text,                   # Text to synthesize
                "reference_text": reference_text, # Text matching the reference audio (required)
                "system_audio": audio_base64,    # Voice reference
                "tts": True,                     # Enable direct TTS mode
                "format": "wav",
                "temperature": 0.7,              # Optional: Control variability
                "normalize": True                # Optional: Improve pronunciation of numbers
            }
        },
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    return response.json()
```

### Reference Audio Requirements

1. Format: WAV file at 44.1kHz
2. Duration: 5-30 seconds (max 90 seconds)
3. Quality: Clean mono audio
4. Content: Clear speech without background noise

## Streaming Audio

```python
import pyaudio
import wave

def stream_audio(text, reference_audio=None):
    payload = {
        "input": {
            "message": text,
            "streaming": True,
            "format": "wav"  # Streaming only supports WAV
        }
    }
    
    if reference_audio:
        payload["input"]["system_audio"] = encode_audio_file(reference_audio)

    response = requests.post(
        ENDPOINT,
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
        stream=True
    )

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                   channels=1,
                   rate=44100,
                   output=True)

    try:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
```

## Using the Test Client

The repository includes a test client for easier interaction with the Fish Speech API:

```bash
# Test with Agent mode (conversational)
python runpod/test_client.py --endpoint YOUR_ENDPOINT_ID --api-key YOUR_API_KEY \
  --message "Hello, how are you today?" --system-audio path/to/reference.wav

# Test with direct TTS mode (recommended for pure voice cloning)
python runpod/test_client.py --endpoint YOUR_ENDPOINT_ID --api-key YOUR_API_KEY \
  --reference-text "Text matching the reference audio" \
  --message "Text to be spoken in the reference voice" \
  --system-audio path/to/reference.wav \
  --tts

# Test locally
python runpod/test_client.py --local --message "Hello world" --system-audio path/to/reference.wav
```

## Advanced Configuration

You can configure the server to start in either "agent" or "tts" mode as the default:

```bash
# Start in TTS mode (optimized for voice cloning)
MODE=tts ./entrypoint.sh

# Start in agent mode (for conversational AI)
MODE=agent ./entrypoint.sh  # or just ./entrypoint.sh
```

Even when running in one mode, you can still access the other mode's functionality by setting the `tts` parameter in your requests.
    finally:
        stream.close()
        p.terminate()
```

## Conversation History

The API maintains conversation state and VQ codes for consistent voice across exchanges:

```python
conversation = [
    {
        "role": "system",
        "content": "You are a helpful voice assistant."
    },
    {
        "role": "user",
        "content": "Hello!"
    }
]

response = requests.post(
    ENDPOINT,
    json={
        "input": {
            "message": "How are you?",
            "conversation_id": "session-123",
            "streaming": False,
            "format": "wav"
        }
    },
    headers={"Authorization": f"Bearer {API_KEY}"}
)

# Response includes conversation history
history = response.json()["output"]["history"]
```

## Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| message | string | Yes (Agent) | Input text for agent mode |
| text | string | Yes (TTS) | Text to synthesize in TTS mode |
| reference_text | string | Yes (TTS) | Text that matches the reference audio content |
| system_message | string | No | Assistant behavior prompt |
| tts | bool | No | Enable direct text-to-speech synthesis (default: false) |
| system_audio | string | Yes (TTS) | Base64 reference audio for voice cloning |
| streaming | bool | No | Enable streaming (default: true) |
| format | string | No | 'wav' (streaming)/'mp3'/'flac' |
| conversation_id | string | No | Session identifier |
| temperature | float | No | Sampling randomness (0.7) |
| top_p | float | No | Nucleus sampling (0.7) |
| chunk_length | int | No | Generation chunk size (200) |
| max_new_tokens | int | No | Max tokens to generate (0) |
| repetition_penalty | float | No | Repetition control (1.2) |

## Error Handling

The handler provides detailed error information:

```python
def safe_request(func):
    try:
        response = func()
        if response.status_code == 200:
            result = response.json()
            if result.get("output", {}).get("status") == "error":
                print(f"API Error: {result['output']['error']}")
                return None
            return result
        else:
            print(f"HTTP Error {response.status_code}")
            error_details = response.json()
            print(f"Details: {error_details.get('output', {}).get('error')}")
    except Exception as e:
        print(f"Request failed: {str(e)}")
    return None
```

Common error codes:
- 400: Invalid input (text too long, bad format)
- 413: Reference audio too large
- 503: Network/server error

## Rate Limits

- `/run` (Async): 1000 requests/10s, 200 concurrent
- `/runsync`: 2000 requests/10s, 400 concurrent
- `/status`, `/stream`: 2000 requests/10s, 400 concurrent
- `/cancel`: 100 requests/10s, 20 concurrent
- `/purge-queue`: 2 requests/10s

Note: Results must be retrieved within 30 minutes.

## Best Practices

1. **Voice Cloning**
   - Use high-quality reference audio (5-30s)
   - Ensure clean speech without background noise
   - Test with shorter samples first
   - Consider caching references with use_memory_cache

2. **Performance**
   - Use streaming for real-time responses
   - Keep messages under max_text_length
   - Use appropriate chunk_length for your needs
   - Monitor response times and adjust parameters

3. **Error Recovery**
   - Implement retry logic for network issues
   - Handle disconnects in streaming mode
   - Cache successful responses locally
   - Check handler status responses

## Support

For issues:
1. Check endpoint health
2. Verify input parameters match documentation
3. Ensure audio meets format requirements
4. Check error responses for details
5. File issues with example code and error traces
