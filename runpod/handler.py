import os
import runpod
from typing import Dict, Any
from loguru import logger

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

class RunPodHandler:
    def __init__(self):
        # Initialize state
        self.agent = FishE2EAgent()
        self.conversations = {}
        
        # Test connection to API server
        logger.info("Testing API server connection...")
        try:
            health = self.agent.client.get(f"{self.agent.vqgan_url}/v1/health")
            if health.status_code == 200:
                logger.info("API server is ready")
            else:
                raise RuntimeError("API server returned error status")
        except Exception as e:
            logger.error(f"API server connection failed: {str(e)}")
            raise

    async def handler(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RunPod serverless requests using FishE2EAgent"""
        try:
            job_id = job.get('id', 'unknown')
            input_data = job.get('input', {})
            
            if not input_data:
                raise ValueError("No input provided")

            # Get conversation state
            conversation_id = input_data.get('conversation_id')
            state = self.conversations.get(conversation_id, ChatState())
            
            # Get input parameters
            text_input = input_data.get('message')
            if not text_input:
                raise ValueError("No message provided")
                
            sys_text = input_data.get('system_message')
            sys_audio = input_data.get('system_audio')  # Base64 audio if provided
            
            # Process system message if provided
            if sys_text and not state.added_systext:
                state.added_systext = True
                state.conversation.append(
                    ServeMessage(
                        role="system",
                        parts=[ServeTextPart(text=sys_text)]
                    )
                )
            
            # Process user input
            state.conversation.append(
                ServeMessage(
                    role="user",
                    parts=[ServeTextPart(text=text_input)]
                )
            )
            
            # Stream responses from agent
            text_response = ""
            audio_data = None
            
            async for event in self.agent.stream(
                sys_audio_data=sys_audio,
                user_audio_data=None,  # Only handling text for now
                sample_rate=44100,
                num_channels=1,
                chat_ctx={
                    "messages": state.conversation,
                    "added_sysaudio": state.added_sysaudio,
                },
            ):
                if event.type == FishE2EEventType.TEXT_SEGMENT:
                    text_response += event.text
                elif event.type == FishE2EEventType.SPEECH_SEGMENT:
                    audio_data = event.frame.data
            
            # Store updated conversation if ID provided
            if conversation_id:
                self.conversations[conversation_id] = state
            
            # Prepare result
            result = {"text": text_response}
            if audio_data is not None:
                result["audio"] = audio_data.tobytes()
                
            return {
                "id": job_id,
                "status": "success",
                "output": result
            }

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            return {
                "id": job_id,
                "status": "error",
                "error": str(e)
            }

def main():
    logger.info("Initializing RunPod handler...")
    handler = RunPodHandler()
    
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({
        "handler": handler.handler
    })

if __name__ == "__main__":
    main()
