import re

import gradio as gr
import numpy as np

from .fish_e2e import FishE2EAgent, FishE2EEventType
from .schema import ServeMessage, ServeTextPart, ServeVQPart


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


def clear_fn():
    return [], ChatState(), None, None, None


async def process_audio_input(
    sys_audio_input, sys_text_input, audio_input, state: ChatState, text_input: str
):
    if audio_input is None and not text_input:
        raise gr.Error("No input provided")

    agent = FishE2EAgent()  # Create new agent instance for each request

    # Convert audio input to numpy array
    if isinstance(audio_input, tuple):
        sr, audio_data = audio_input
    elif text_input:
        sr = 44100
        audio_data = None
    else:
        raise gr.Error("Invalid audio format")

    if isinstance(sys_audio_input, tuple):
        sr, sys_audio_data = sys_audio_input
    elif text_input:
        sr = 44100
        sys_audio_data = None
    else:
        raise gr.Error("Invalid audio format")

    def append_to_chat_ctx(
        part: ServeTextPart | ServeVQPart, role: str = "assistant"
    ) -> None:
        if not state.conversation or state.conversation[-1].role != role:
            state.conversation.append(ServeMessage(role=role, parts=[part]))
        else:
            state.conversation[-1].parts.append(part)

    if state.added_systext is False and sys_text_input:
        state.added_systext = True
        append_to_chat_ctx(ServeTextPart(text=sys_text_input), role="system")
    if text_input:
        append_to_chat_ctx(ServeTextPart(text=text_input), role="user")
        audio_data = None

    result_audio = b""
    async for event in agent.stream(
        sys_audio_data,
        audio_data,
        sr,
        1,
        chat_ctx={
            "messages": state.conversation,
            "added_sysaudio": state.added_sysaudio,
        },
    ):
        if event.type == FishE2EEventType.USER_CODES:
            append_to_chat_ctx(ServeVQPart(codes=event.vq_codes), role="user")
        elif event.type == FishE2EEventType.SPEECH_SEGMENT:
            result_audio += event.frame.data
            np_audio = np.frombuffer(result_audio, dtype=np.int16)
            append_to_chat_ctx(ServeVQPart(codes=event.vq_codes))

            yield state.get_history(), (44100, np_audio), None, None
        elif event.type == FishE2EEventType.TEXT_SEGMENT:
            append_to_chat_ctx(ServeTextPart(text=event.text))
            if result_audio:
                np_audio = np.frombuffer(result_audio, dtype=np.int16)
                yield state.get_history(), (44100, np_audio), None, None
            else:
                yield state.get_history(), None, None, None

    np_audio = np.frombuffer(result_audio, dtype=np.int16)
    yield state.get_history(), (44100, np_audio), None, None


async def process_text_input(
    sys_audio_input, sys_text_input, state: ChatState, text_input: str
):
    async for event in process_audio_input(
        sys_audio_input, sys_text_input, None, state, text_input
    ):
        yield event


def create_demo():
    with gr.Blocks() as demo:
        state = gr.State(ChatState())

        with gr.Row():
            # Left column (70%) for chatbot
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=600,
                    type="messages",
                )

            # Right column (30%) for controls
            with gr.Column(scale=3):
                sys_audio_input = gr.Audio(
                    sources=["upload"],
                    type="numpy",
                    label="Give a timbre for your assistant",
                )
                sys_text_input = gr.Textbox(
                    label="What is your assistant's role?",
                    value='您是由 Fish Audio 设计的语音助手，提供端到端的语音交互，实现无缝用户体验。首先转录用户的语音，然后使用以下格式回答："Question: [用户语音]\n\nResponse: [你的回答]\n"。',
                    type="text",
                )
                audio_input = gr.Audio(
                    sources=["microphone"], type="numpy", label="Speak your message"
                )

                text_input = gr.Textbox(label="Or type your message", type="text")

                output_audio = gr.Audio(label="Assistant's Voice", type="numpy")

                send_button = gr.Button("Send", variant="primary")
                clear_button = gr.Button("Clear")

        # Event handlers
        audio_input.stop_recording(
            process_audio_input,
            inputs=[sys_audio_input, sys_text_input, audio_input, state, text_input],
            outputs=[chatbot, output_audio, audio_input, text_input],
            show_progress=True,
        )

        send_button.click(
            process_text_input,
            inputs=[sys_audio_input, sys_text_input, state, text_input],
            outputs=[chatbot, output_audio, audio_input, text_input],
            show_progress=True,
        )

        text_input.submit(
            process_text_input,
            inputs=[sys_audio_input, sys_text_input, state, text_input],
            outputs=[chatbot, output_audio, audio_input, text_input],
            show_progress=True,
        )

        clear_button.click(
            clear_fn,
            inputs=[],
            outputs=[chatbot, state, audio_input, output_audio, text_input],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
