# Fish Speech RunPod Deployment

This guide explains how to deploy the Fish Speech Agent API on RunPod serverless.

## Project Structure

```
fish-speech/
├── fish_speech/         # Core Fish Speech package
├── tools/              # Server and utility tools
│   ├── server/         # Server implementation
│   │   ├── agent/     # Agent generation code
│   │   ├── model_manager.py
│   │   └── ...
├── runpod/            # RunPod serverless implementation
│   ├── handler.py     # RunPod handler
│   └── README.md      # This guide
├── Dockerfile.runpod  # RunPod container config
├── pyproject.toml     # Project metadata
├── requirements.txt   # Dependencies
└── setup.py          # Package setup
```

## Prerequisites

- RunPod account with GPU credits
- Downloaded models:
  - `checkpoints/fish-agent-v0.1-3b/` - Fish Agent model
  - `checkpoints/fish-speech-1.4/` - Fish Speech model
  - Required GPU: 16GB+ (RTX 4090 recommended)

## Setup Steps

1. Download models (if not already in checkpoints):
```bash
mkdir -p checkpoints
huggingface-cli download fishaudio/fish-agent-v0.1-3b --local-dir checkpoints/fish-agent-v0.1-3b
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

2. Build Docker image:
```bash
cd fish-speech
docker build -f Dockerfile.runpod -t your-registry/fish-speech:latest .
```

3. Push to Docker registry:
```bash
docker push your-registry/fish-speech:latest
```

4. Create RunPod endpoint:
   - Go to RunPod Console: https://runpod.io/console/serverless
   - Click "New Endpoint"
   - Configure:
     - Name: `fish-speech`
     - Docker Image: `your-registry/fish-speech:latest`
     - GPU: RTX 4090 (recommended)
     - Memory: 16GB
     - Container Disk: 20GB

## API Usage

### Basic Chat

```python
import runpod

endpoint = "your-endpoint-id"
response = await runpod.run(endpoint, {
    "input": {
        "message": "Hello, how are you?",
        "conversation_id": "test-1",  # Optional: for context
        "temperature": 0.7,           # Optional: 0.1-1.0
        "max_tokens": 1000            # Optional: max response length
    }
})

print(response["output"]["text"])
```

### Voice Clone Chat

```python
import runpod
import base64

# Load reference audio (10-30 seconds, clear voice)
with open("reference.wav", "rb") as f:
    reference_audio = base64.b64encode(f.read()).decode()

response = await runpod.run(endpoint, {
    "input": {
        "message": "Tell me a story",
        "reference_audio": reference_audio,
        "conversation_id": "test-2",
        "temperature": 0.7,
        "max_tokens": 1000
    }
})

# Get text and audio
text = response["output"]["text"]
if "audio" in response["output"]:
    import numpy as np
    audio = np.array(response["output"]["audio"])
    # Save as WAV file using soundfile or similar
```

## Response Format

### Success
```python
{
    "id": "job-id",
    "status": "success",
    "output": {
        "text": "Agent's response",
        "audio": [...],  # Optional: if voice cloning used
    }
}
```

### Error
```python
{
    "id": "job-id",
    "status": "error",
    "error": "Error message"
}
```

## Performance Notes

1. First Request:
   - Model compilation on first run (~30s)
   - Subsequent requests are faster
   - RTX 4090: ~95 tokens/second

2. Voice Cloning:
   - Reference audio: 10-30 seconds
   - Clear voice, minimal noise
   - WAV format recommended

3. Memory Usage:
   - Base: ~8GB
   - Peak: ~14GB
   - With voice cloning: ~16GB

## Environment Variables

```bash
# Required
MODEL_PATH=/app/checkpoints
LLAMA_CHECKPOINT_PATH=/app/checkpoints/fish-agent-v0.1-3b
DECODER_CHECKPOINT_PATH=/app/checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth

# Optional
DECODER_CONFIG_NAME=base  # Model configuration
MODE=agent               # Service mode
HALF=true               # Use half precision
COMPILE=true            # Enable torch compilation
ASR_ENABLED=false       # Disable ASR for agent mode
```

## Common Issues

1. GPU Memory:
   - Error: `RuntimeError: CUDA out of memory`
   - Solution: Reduce batch size or use larger GPU

2. Model Loading:
   - Error: `FileNotFoundError: Model not found`
   - Solution: Verify model paths and files

3. Voice Generation:
   - Error: `Invalid reference audio`
   - Solution: Check audio format and length

## Support

- RunPod Issues: [RunPod Support](https://runpod.io/support)
- Fish Speech Issues: [GitHub Repository](https://github.com/fishaudio/fish-speech/issues)
