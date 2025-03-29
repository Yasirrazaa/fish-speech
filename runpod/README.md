# Fish Speech RunPod API

This guide explains how to use the Fish Speech API in your applications across different programming languages.

## API Overview

Fish Speech provides a voice-enabled AI assistant that can generate both text and speech responses. The API is designed to be simple to use while offering advanced features like voice cloning and conversation history.

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
        "conversation_id": "optional-conversation-identifier",
        "temperature": 0.7,
        "max_tokens": 1000
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
        "audio": [array of audio samples],
        "history": [
            {"role": "user", "content": "Your message"},
            {"role": "assistant", "content": "Assistant's response"}
        ]
    }
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

# API endpoint URL
ENDPOINT = "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run"

def chat_with_fish_speech(message, system_message=None, conversation_id=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {YOUR_API_KEY}"
    }
    
    payload = {
        "input": {
            "message": message
        }
    }
    
    # Add optional parameters if provided
    if system_message:
        payload["input"]["system_message"] = system_message
    if conversation_id:
        payload["input"]["conversation_id"] = conversation_id
        
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    result = response.json()
    
    if result.get("status") == "success":
        return result["output"]
    else:
        raise Exception(f"API Error: {result.get('error', 'Unknown error')}")

# Example usage
response = chat_with_fish_speech(
    message="Tell me a short story about a robot",
    system_message="You are a creative storyteller"
)

print(response["text"])

# If audio is present, save it to a file
if "audio" in response:
    audio_data = np.array(response["audio"])
    sf.write("response.wav", audio_data, 44100)
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

const ENDPOINT = 'https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run';
const API_KEY = '{YOUR_API_KEY}';

async function chatWithFishSpeech(message, systemMessage = null, conversationId = null) {
    const payload = {
        input: {
            message: message
        }
    };
    
    if (systemMessage) payload.input.system_message = systemMessage;
    if (conversationId) payload.input.conversation_id = conversationId;
    
    try {
        const response = await axios.post(ENDPOINT, payload, {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            }
        });
        
        if (response.data.status === 'success') {
            return response.data.output;
        } else {
            throw new Error(`API Error: ${response.data.error || 'Unknown error'}`);
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
            'You are a helpful weather assistant'
        );
        
        console.log(response.text);
        
        // If audio is present, save it to a file
        if (response.audio) {
            const Float32Array = new Float32Array(response.audio);
            // Note: You'll need a library to save audio data to a file in Node.js
            // This is just a placeholder for the concept
            fs.writeFileSync('response.raw', Buffer.from(Float32Array.buffer));
            console.log('Audio saved to response.raw');
        }
    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
```

### cURL (Command Line)

```bash
curl -X POST "https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer {YOUR_API_KEY}" \
     -d '{
         "input": {
             "message": "Tell me a joke",
             "system_message": "You are a funny assistant",
             "conversation_id": "joke-session"
         }
     }'
```

### Using with voice input

To use voice cloning features, you need to provide a reference audio file encoded in base64:

```python
import base64

# Read audio file and encode as base64
with open("reference_voice.wav", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

payload = {
    "input": {
        "message": "Please say this in the reference voice style",
        "system_audio": audio_base64  # Base64-encoded audio
    }
}

# Then send this payload to the API as shown in the examples above
```

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| message | string | Yes | The text message to send to the assistant |
| system_message | string | No | Instructions to control assistant behavior |
| system_audio | string | No | Base64-encoded audio file for voice cloning |
| conversation_id | string | No | Unique identifier to maintain conversation context |
| temperature | float | No | Sampling temperature (0.1-1.0), controls randomness |
| max_tokens | integer | No | Maximum tokens in response |

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
    --output-dir outputs
```

## Limitations and Best Practices

1. First requests may take longer (10-15 seconds) as the model loads
2. Voice cloning works best with high-quality audio samples (5-10 seconds of clear speech)
3. Keep conversation IDs unique and persistent for continuous conversations
4. Use system messages to guide the assistant's behavior and voice style
5. For best performance, use concise messages and keep conversation history short

## Getting Help

If you encounter issues with the API, please check:

1. Your API key and endpoint ID are correct
2. Your request format matches the required structure
3. For any persistent issues, please file a bug on our GitHub repository

## Advanced Features

### Streaming Responses

For applications requiring real-time responses, consider implementing a polling mechanism since the API is asynchronous:

```python
import time

def submit_request(message):
    # Submit initial request
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    return response.json()["id"]  # Get the request ID

def check_status(request_id):
    status_url = f"https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/status/{request_id}"
    response = requests.get(status_url, headers=headers)
    return response.json()

# Submit request and poll for completion
request_id = submit_request("Hello, how are you?")
while True:
    status = check_status(request_id)
    if status["status"] == "COMPLETED":
        print("Response:", status["output"]["text"])
        break
    elif status["status"] == "FAILED":
        print("Request failed:", status["error"])
        break
    time.sleep(1)  # Wait before checking again
```
