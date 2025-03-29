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
