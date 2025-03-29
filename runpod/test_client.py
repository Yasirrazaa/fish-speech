#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

import soundfile as sf
import numpy as np
import runpod

class FishSpeechClient:
    def __init__(self, endpoint="local"):
        self.endpoint = endpoint

    async def chat(
        self, 
        message: str, 
        reference_audio: str = None,
        conversation_id: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Chat with the Fish Speech API
        
        Args:
            message: Text message to send
            reference_audio: Path to WAV file for voice cloning
            conversation_id: Optional ID to maintain context
            temperature: Sampling temperature (0.1-1.0)
            max_tokens: Maximum response length
        """
        input_data = {
            "message": message,
            "conversation_id": conversation_id,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add reference audio if provided
        if reference_audio:
            with open(reference_audio, 'rb') as f:
                input_data["reference_audio"] = base64.b64encode(f.read()).decode()
        
        response = await runpod.run(self.endpoint, {
            "input": input_data
        })
        
        if response["status"] == "error":
            raise Exception(response["error"])
            
        return response["output"]

async def main():
    parser = argparse.ArgumentParser(description="Fish Speech API Client")
    parser.add_argument(
        "--endpoint", 
        default="local",
        help="RunPod endpoint ID (use 'local' for local testing)"
    )
    parser.add_argument(
        "--message", 
        type=str, 
        required=True,
        help="Message to send"
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        help="Path to reference audio file (WAV) for voice cloning"
    )
    parser.add_argument(
        "--conversation-id",
        type=str,
        help="Conversation ID for context"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.1-1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum response length"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = FishSpeechClient(args.endpoint)
    
    try:
        print("\nSending request...")
        output = await client.chat(
            message=args.message,
            reference_audio=args.reference_audio,
            conversation_id=args.conversation_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Save text response
        text_file = output_dir / "response.txt"
        with open(text_file, 'w') as f:
            f.write(output["text"])
        print(f"\nText response saved to: {text_file}")
        print("\nResponse:", output["text"])
        
        # Save audio if present
        if "audio" in output:
            audio_file = output_dir / "response.wav"
            audio_data = np.array(output["audio"])
            sf.write(audio_file, audio_data, 44100)
            print(f"Audio response saved to: {audio_file}")
            
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
