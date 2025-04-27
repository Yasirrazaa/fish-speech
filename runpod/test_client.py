#!/usr/bin/env python3
"""
Test client for Fish Speech RunPod endpoint
"""
import os
import sys
import json
import time
import base64
import argparse
import requests
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fish Speech RunPod Test Client")
    
    # Add arguments
    parser.add_argument("--endpoint", type=str, help="RunPod endpoint ID")
    parser.add_argument("--api-key", type=str, help="RunPod API key")
    parser.add_argument("--message", type=str, default="Hello, how are you today?", help="Message to send")
    parser.add_argument("--system", type=str, help="System message")
    parser.add_argument("--system-audio", type=str, help="Path to reference audio file for voice cloning")
    parser.add_argument("--reference-text", type=str, help="Text that matches the reference audio (for TTS mode)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--conversation-id", type=str, help="Conversation ID for continuing a conversation")
    parser.add_argument("--timeout", type=int, default=1000, help="Timeout in seconds")
    parser.add_argument("--local", action="store_true", help="Test with local handler.py instead of RunPod endpoint")
    parser.add_argument("--tts", action="store_true", help="Force using TTS mode for direct voice cloning")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming response")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "mp3", "flac"], help="Output audio format")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Fish Speech RunPod Test Client")
    print(f"Output directory: {output_dir}")
    
    # Check if we're testing locally
    if args.local:
        print("🧪 Testing with local handler...")
        # Set environment variable to indicate we're doing local testing
        os.environ["RUNPOD_LOCAL_TEST"] = "1"
        
        # Create a test input file
        test_input = {
            "id": "local_test",
            "input": {
                "message": args.message
            }
        }
        
        if args.system:
            test_input["input"]["system_message"] = args.system
        
        if args.conversation_id:
            test_input["input"]["conversation_id"] = args.conversation_id
        
        if args.system_audio:
            with open(args.system_audio, "rb") as f:
                audio_data = f.read()
                test_input["input"]["system_audio"] = base64.b64encode(audio_data).decode("utf-8")
        
        # For TTS mode, ensure we have all required parameters
        if args.tts:
            if not args.reference_text or not args.system_audio:
                print("❌ Error: TTS mode requires both --reference-text and --system-audio")
                return
            test_input["input"]["reference_text"] = args.reference_text
            test_input["input"]["text"] = args.message  # Use message as the text to synthesize
            test_input["input"]["tts"] = True
        
        # Add streaming flag if specified
        if args.streaming:
            test_input["input"]["streaming"] = True
        
        if args.format:
            test_input["input"]["format"] = args.format

        
        # Save test input to file
        with open("test_input.json", "w") as f:
            json.dump(test_input, f, indent=2)
        
        # Run the handler script directly
        print("🔄 Running handler.py...")
        import subprocess
        subprocess.run([sys.executable, "runpod/handler.py"])
        
        # Check if output files were created
        # Check output files with proper format handling
        text_file = output_dir / "response.txt"
        audio_file = output_dir / f"response.{args.format}"
        
        if text_file.exists():
            print(f"✅ Text response saved to {text_file}")
            with open(text_file, "r") as f:
                print(f"📝 Response: {f.read()[:100]}...")
        
        if audio_file.exists():
            print(f"🔊 Audio response saved to {audio_file}")
        
        return
    
    # Validate API key and endpoint for RunPod testing
    if not args.endpoint or not args.api_key:
        print("❌ Error: --endpoint and --api-key are required for RunPod testing")
        parser.print_help()
        return
    
    # Prepare the payload
    payload = {
        "input": {
            "message": args.message
        }
    }
    
    if args.system:
        payload["input"]["system_message"] = args.system
    
    if args.conversation_id:
        payload["input"]["conversation_id"] = args.conversation_id
    
    if args.system_audio:
        with open(args.system_audio, "rb") as f:
            audio_data = f.read()
            payload["input"]["system_audio"] = base64.b64encode(audio_data).decode("utf-8")
            
    # For TTS mode, ensure we have all required parameters
    if args.tts:
        if not args.reference_text or not args.system_audio:
            print("❌ Error: TTS mode requires both --reference-text and --system-audio")
            return
        payload["input"]["reference_text"] = args.reference_text
        payload["input"]["text"] = args.message  # Use message as the text to synthesize
        payload["input"]["tts"] = True

    # Add streaming flag if specified
    if args.streaming:
        payload["input"]["streaming"] = True
    
    if args.format:
        payload["input"]["format"] = args.format
    
    # API endpoint
    endpoint_url = f"https://api.runpod.ai/v2/{args.endpoint}/run"
    status_url = f"https://api.runpod.ai/v2/{args.endpoint}/status"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"
    }
    
    print(f"🔄 Sending request to endpoint: {args.endpoint}")
    print(f"📤 Message: {args.message}")
    
    try:
        # Submit job
        response = requests.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        if "id" not in response_data:
            print(f"❌ Error: No job ID in response: {response_data}")
            return
        
        job_id = response_data["id"]
        print(f"✅ Job submitted successfully with ID: {job_id}")
        
        # Poll for results
        start_time = time.time()
        while True:
            if time.time() - start_time > args.timeout:
                print(f"❌ Timeout after {args.timeout} seconds")
                return
            
            # Check job status
            status_response = requests.get(f"{status_url}/{job_id}", headers=headers)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            if status_data.get("status") == "COMPLETED":
                print("✅ Job completed successfully!")
                output = status_data.get("output", {})
                
                # Save text response
                if "text" in output:
                    with open(output_dir / "response.txt", "w") as f:
                        f.write(output["text"])
                    print(f"📝 Text response: {output['text'][:100]}...")
                    print(f"📄 Full response saved to {output_dir / 'response.txt'}")
                
                # Save audio response if present (handle both audio_base64 and audio keys)
                audio_content = output["output"].get("audio_base64") or output['output'].get("audio")
                if audio_content:
                    audio_bytes = base64.b64decode(audio_content)
                    output_file = output_dir / f"response.{args.format}"
                    with open(output_file, "wb") as f:
                        f.write(audio_bytes)
                    print(f"🔊 Audio response saved to {output_file}")
                
                # Save full response
                with open(output_dir / "response.json", "w") as f:
                    json.dump(status_data, f, indent=2)
                
                break
                
            elif status_data.get("status") == "FAILED":
                print(f"❌ Job failed: {status_data.get('error', 'Unknown error')}")
                break
                
            else:
                print(f"🔄 Job status: {status_data.get('status')} - waiting...")
                time.sleep(5)
    
    except requests.RequestException as e:
        print(f"❌ Request error: {e}")
    except KeyboardInterrupt:
        print("❌ Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
