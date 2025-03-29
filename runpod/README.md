# Fish Speech RunPod API

This guide explains how to use the Fish Speech API in your applications across different programming languages.

## API Overview

Fish Speech provides a voice-enabled AI assistant that can generate both text and speech responses using an end-to-end (E2E) voice synthesis approach. The API is designed to be simple to use while offering advanced features like voice cloning, conversation memory via VQ codes, and streaming responses.

## Getting Your API Key

To use the Fish Speech API, you'll need a RunPod API key:

1. Log in to your [RunPod account](https://runpod.io)
2. Click on your profile icon in the top-right corner
3. Select "API Keys" from the menu
4. Click "Create API Key"
5. Give your key a name (e.g., "Fish-Speech-API")
6. Copy the generated API key (it will only be shown once)

Keep your API key secure and don't share it publicly. You'll also need your endpoint ID, which is visible on your RunPod dashboard after creating your serverless endpoint.

## Input Format

All requests to the API should follow this JSON structure:

```json
{
    "input": {
        "message": "Your message text here",
        "system_message": "Optional system prompt to define assistant behavior",
        "system_audio": "Optional base64-encoded audio for voice cloning",
        "conversation_id": "optional-conversation-identifier",
        "streaming": true,
        "format": "wav",
        "temperature": 0.7,
        "max_new_tokens": 512,
        "max_text_length": 1000
    }
}
```

A sample input file is provided at `runpod/test_input.json`.

## Response Format

The API returns responses in the following format:

```json
{
    "id": "request-id",
    "status": "success",
    "output": {
        "text": "The text response from the assistant",
        "audio_base64": "base64-encoded audio data",
        "audio_format": "wav",
        "sample_rate": 44100,
        "history": [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Your message"},
            {"role": "assistant", "content": "Assistant's response <audio 2.5s>"}
        ],
        "vq_codes": [[...]] // For advanced applications maintaining voice consistency
    }
}
```

For streaming responses, the API will send chunks in this format:
```json
{
    "type": "text",
    "content": "Text segment"
}
```
or
```json
{
    "type": "audio",
    "content": "base64-encoded audio chunk"
}
```

## Code Examples

### Python

```python
import requests
import json
import base64
import numpy as np
import soundfile as sf
import io

# API endpoint URL
ENDPOINT = "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run"

def chat_with_fish_speech(message, system_message=None, system_audio=None, conversation_id=None, streaming=False):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {YOUR_API_KEY}"
    }
    
    payload = {
        "input": {
            "message": message,
            "streaming": streaming,
            "format": "wav",
            "temperature": 0.7,
            "max_new_tokens": 512
        }
    }
    
    # Add optional parameters if provided
    if system_message:
        payload["input"]["system_message"] = system_message
    if system_audio:
        with open(system_audio, "rb") as f:
            audio_bytes = f.read()
            payload["input"]["system_audio"] = base64.b64encode(audio_bytes).decode('utf-8')
    if conversation_id:
        payload["input"]["conversation_id"] = conversation_id
        
    if streaming:
        # Handle streaming response
        response = requests.post(ENDPOINT, headers=headers, json=payload, stream=True)
        for chunk in response.iter_lines():
            if chunk:
                data = json.loads(chunk.decode('utf-8').replace('data: ', ''))
                if data['type'] == 'text':
                    print(data['content'], end='', flush=True)
                elif data['type'] == 'audio':
                    # Process audio chunk
                    audio_bytes = base64.b64decode(data['content'])
                    # Play audio or save as needed
    else:
        # Handle regular response
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        result = response.json()
        
        if result.get("status") == "success":
            output = result["output"]
            
            # If audio is present, decode and save it
            if "audio_base64" in output:
                audio_bytes = base64.b64decode(output["audio_base64"])
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                sf.write("response.wav", audio_data, output["sample_rate"])
                
            return output
        else:
            raise Exception(f"API Error: {result.get('error', 'Unknown error')}")

# Example usage
response = chat_with_fish_speech(
    message="Tell me a short story about a robot",
    system_message="You are a creative storyteller with a warm, friendly voice",
    system_audio="reference_voice.wav"  # Optional: for voice cloning
)

print(response["text"])
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

const ENDPOINT = 'https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run';
const API_KEY = '{YOUR_API_KEY}';

async function chatWithFishSpeech(message, options = {}) {
    const { systemMessage, systemAudio, conversationId, streaming = false } = options;
    
    const payload = {
        input: {
            message: message,
            streaming: streaming,
            format: "wav",
            temperature: 0.7,
            max_new_tokens: 512
        }
    };
    
    if (systemMessage) payload.input.system_message = systemMessage;
    if (conversationId) payload.input.conversation_id = conversationId;
    
    // Add system audio if provided
    if (systemAudio) {
        const audioData = fs.readFileSync(systemAudio);
        payload.input.system_audio = audioData.toString('base64');
    }
    
    try {
        if (streaming) {
            const response = await axios.post(ENDPOINT, payload, {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEY}`
                },
                responseType: 'stream'
            });
            
            response.data.on('data', (chunk) => {
                const data = JSON.parse(chunk.toString().replace('data: ', ''));
                if (data.type === 'text') {
                    process.stdout.write(data.content);
                } else if (data.type === 'audio') {
                    // Process audio chunk
                    const audioBuffer = Buffer.from(data.content, 'base64');
                    // Play audio or save as needed
                }
            });
            
            return new Promise((resolve) => {
                response.data.on('end', () => {
                    resolve();
                });
            });
        } else {
            const response = await axios.post(ENDPOINT, payload, {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_KEY}`
                }
            });
            
            if (response.data.status === 'success') {
                const output = response.data.output;
                
                // If audio is present, save it to a file
                if (output.audio_base64) {
                    const audioBuffer = Buffer.from(output.audio_base64, 'base64');
                    fs.writeFileSync('response.wav', audioBuffer);
                    console.log('Audio saved to response.wav');
                }
                
                return output;
            } else {
                throw new Error(`API Error: ${response.data.error || 'Unknown error'}`);
            }
        }
    } catch (error) {
        console.error('Request failed:', error);
        throw error;
    }
}

// Example usage
async function main() {
    try {
        const response = await chatWithFishSpeech(
            'What is the weather like today?',
            {
                systemMessage: 'You are a helpful weather assistant',
                systemAudio: 'reference_voice.wav',  // Optional: for voice cloning
                streaming: false
            }
        );
        
        console.log(response.text);
    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
```

### cURL (Command Line)

```bash
# For a basic request without streaming
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer {YOUR_API_KEY}" \
     -d '{
         "input": {
             "message": "Tell me a joke",
             "system_message": "You are a funny assistant",
             "conversation_id": "joke-session",
             "streaming": false,
             "format": "wav",
             "temperature": 0.7
         }
     }'

# For streaming response (requires a client that can handle streamed responses)
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer {YOUR_API_KEY}" \
     -d '{
         "input": {
             "message": "Tell me a joke",
             "system_message": "You are a funny assistant",
             "streaming": true,
             "format": "wav"
         }
     }'
```

### Using with voice input (Voice Cloning)

To use voice cloning features, you need to provide a reference audio file encoded in base64:

```python
import base64

# Read audio file and encode as base64
with open("reference_voice.wav", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

payload = {
    "input": {
        "message": "Please say this in the reference voice style",
        "system_audio": audio_base64,  # Base64-encoded audio for voice cloning
        "system_message": "You are a helpful assistant speaking in the style of the provided voice"
    }
}

# Then send this payload to the API as shown in the examples above
```

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| message | string | Yes | The text message to send to the assistant |
| system_message | string | No | Instructions to control assistant behavior |
| system_audio | string | No | Base64-encoded audio file for voice cloning (max 90 seconds) |
| conversation_id | string | No | Unique identifier to maintain conversation context |
| streaming | boolean | No | Enable streaming response (default: true) |
| format | string | No | Audio format: "wav" (default), "mp3", or "flac" (streaming only supports "wav") |
| temperature | float | No | Sampling temperature (0.1-1.0), controls randomness (default: 0.7) |
| max_new_tokens | integer | No | Maximum tokens in response (default: 512) |
| max_text_length | integer | No | Maximum input text length (default: 1000) |

## Error Handling

The API returns error details in the response when something goes wrong:

```json
{
    "id": "request-id",
    "status": "error",
    "output": {
        "error": "Error message",
        "trace": "Detailed error trace"
    }
}
```

Common error codes:
- 400: Bad request (invalid parameters or input too long)
- 500: Server error (model loading failure or processing error)

## Testing Your Integration

1. Use the provided `test_input.json` file to test your API requests:

```bash
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer {YOUR_API_KEY}" \
     -d @test_input.json
```

2. For local testing, use the included test client:

```bash
python -m runpod.test_client \
    --endpoint your-endpoint-id \
    --api-key your-api-key \
    --message "Hello, world!" \
    --system-message "You are a helpful assistant" \
    --reference-audio reference_voice.wav \
    --output-dir outputs
```

## Technical Implementation

The Fish Speech RunPod handler uses the following components:

1. **FishE2EAgent**: Handles the conversation flow, audio processing, and text generation
2. **VQ Codes**: Vector-quantized tokens that maintain voice consistency across responses
3. **LiveKit AudioFrames**: Used for efficient audio data handling
4. **Streaming Protocol**: Custom protocol for efficiently streaming both text and audio responses

The handler is designed to be resilient with features like:
- Automatic retries for network failures
- Memory management to prevent OOM errors
- Configurable parameters for different use cases

## Limitations and Best Practices

1. First requests may take longer (10-15 seconds) as the model loads
2. Voice cloning works best with high-quality audio samples (5-10 seconds of clear speech)
3. Keep conversation IDs unique and persistent for continuous conversations
4. Use system messages to guide the assistant's behavior and voice style
5. For best performance, use concise messages and keep conversation history short
6. System audio should be clear, contain minimal background noise, and ideally be 5-30 seconds in length

## Getting Help

If you encounter issues with the API, please check:

1. Your API key and endpoint ID are correct
2. Your request format matches the required structure
3. The reference audio is properly encoded and not too large
4. For any persistent issues, please file a bug on our GitHub repository

## Advanced Features

### Conversation State Management

The API uses VQ codes to maintain voice consistency across a conversation. When using the API for ongoing conversations:

1. Always include the same `conversation_id` for related exchanges
2. The system will automatically track voice characteristics using VQ codes
3. For advanced applications, you can extract and store the `vq_codes` from responses

### Custom Voice Creation

To create a custom voice profile:

1. Record 5-30 seconds of high-quality speech
2. Convert to WAV format, 44.1kHz, 16-bit
3. Encode as base64 and include as `system_audio`
4. Use consistent `conversation_id` to maintain this voice in conversations

### Hybrid Development Approach

For development and testing purposes, you can:

1. Run the Fish Speech service locally using Docker (see documentation in `/docs/en/inference.md`)
2. Use the same API format for both local development and RunPod deployment
3. Test features locally before moving to production on RunPod
