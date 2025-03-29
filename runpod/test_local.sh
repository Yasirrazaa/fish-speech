#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Fish Speech RunPod Local Test${NC}"
echo "=============================="

# Check CUDA
echo -e "\n${GREEN}Checking CUDA...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA not available.${NC}"
    exit 1
fi

nvidia-smi
CUDA_AVAILABLE=$?
if [ $CUDA_AVAILABLE -ne 0 ]; then
    echo -e "${RED}Error: CUDA test failed${NC}"
    exit 1
fi

# Check models directory
echo -e "\n${GREEN}Checking models...${NC}"
if [ ! -d "../checkpoints/fish-agent-v0.1-3b" ] || [ ! -d "../checkpoints/fish-speech-1.5" ]; then
    echo -e "${RED}Error: Models not found in checkpoints directory${NC}"
    echo "Please ensure you have:"
    echo "  - checkpoints/fish-agent-v0.1-3b/"
    echo "  - checkpoints/fish-speech-1.5/"
    exit 1
fi

# Build Docker image
echo -e "\n${GREEN}Building Docker image...${NC}"
IMAGE_NAME="fish-speech-test"
docker build -f ../Dockerfile.runpod -t $IMAGE_NAME ..

# Create test script
echo -e "\n${GREEN}Creating test script...${NC}"
mkdir -p test
cat > test/test_api.py << 'EOL'
import runpod
import asyncio
import json
import base64
import numpy as np
import soundfile as sf

async def test_text_chat():
    print("\nTesting text chat...")
    response = await runpod.run(
        "local",  # Use local for testing
        {
            "input": {
                "message": "Hello! How are you?",
                "conversation_id": "test-1",
                "temperature": 0.7,
                "max_tokens": 100  # Small for testing
            }
        }
    )
    print("Response:", json.dumps(response, indent=2))
    return "text" in response.get("output", {})

async def test_voice_chat():
    print("\nTesting voice chat...")
    # Generate test audio
    sample_rate = 44100
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save test audio
    sf.write('test_audio.wav', audio, sample_rate)
    
    # Read and encode audio
    with open('test_audio.wav', 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    response = await runpod.run(
        "local",
        {
            "input": {
                "message": "Tell me a short story",
                "reference_audio": audio_base64,
                "conversation_id": "test-2",
                "temperature": 0.7,
                "max_tokens": 100
            }
        }
    )
    print("Response:", json.dumps(response, indent=2))
    
    # Save output audio if present
    if "audio" in response.get("output", {}):
        audio_data = np.array(response["output"]["audio"])
        sf.write('response_audio.wav', audio_data, sample_rate)
        print("Response audio saved to response_audio.wav")
    
    return "text" in response.get("output", {}) and "audio" in response.get("output", {})

async def main():
    try:
        text_ok = await test_text_chat()
        voice_ok = await test_voice_chat()
        
        if text_ok and voice_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
EOL

# Create test container
echo -e "\n${GREEN}Running test container...${NC}"
docker run -d \
    --name fish-speech-test \
    --gpus all \
    -v $(pwd)/../checkpoints:/app/checkpoints \
    -v $(pwd)/test:/app/test \
    -p 8080:8080 \
    $IMAGE_NAME

# Wait for container startup
echo -e "\n${YELLOW}Waiting for container to start (30s)...${NC}"
sleep 30

# Install test dependencies
echo -e "\n${GREEN}Installing test dependencies...${NC}"
pip install runpod soundfile numpy

# Run tests
echo -e "\n${GREEN}Running tests...${NC}"
cd test
python test_api.py

# Cleanup
echo -e "\n${GREEN}Cleaning up...${NC}"
docker stop fish-speech-test
docker rm fish-speech-test

echo -e "\n${GREEN}Test completed!${NC}"
