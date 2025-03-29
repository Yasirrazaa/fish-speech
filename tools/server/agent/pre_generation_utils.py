import queue

from fish_speech.conversation import Conversation, Message
from fish_speech.models.text2semantic.inference import GenerateRequest
from fish_speech.tokenizer import IM_END_TOKEN


# Add tokenizer compatibility wrapper
def get_token_id_safe(tokenizer, token):
    """
    Safe wrapper to get a token ID from a tokenizer, handling different tokenizer types.
    """
    # Check if the tokenizer has the method directly
    if hasattr(tokenizer, "get_token_id"):
        return tokenizer.get_token_id(token)
    
    # For Hugging Face tokenizers (e.g. Qwen2TokenizerFast)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        return tokenizer.convert_tokens_to_ids(token)
    
    # For tokenizers with direct vocab access
    if hasattr(tokenizer, "vocab") and token in tokenizer.vocab:
        return tokenizer.vocab[token]
    
    # For tiktoken-style tokenizers
    if hasattr(tokenizer, "encode"):
        try:
            token_ids = tokenizer.encode(token)
            if token_ids and len(token_ids) == 1:
                return token_ids[0]
        except:
            pass
    
    # Fallback: Try to find a special token
    if hasattr(tokenizer, "special_tokens_map") and token in tokenizer.special_tokens_map:
        special_token = tokenizer.special_tokens_map[token]
        return tokenizer.convert_tokens_to_ids(special_token)
    
    raise ValueError(f"Could not get token ID for '{token}' with the current tokenizer ({type(tokenizer).__name__})")


def prepare_messages(request, tokenizer, config):
    """
    Reorganise the provided list of messages into a conversation.
    Encode the conversation for inference.
    """
    # Convert the messages to ConversationMessage objects
    messages = [msg.to_conversation_message() for msg in request.messages]

    if len(messages) < 1:
        raise ValueError("At least one message is required")

    # Check the last message to determine the next step
    last_role = messages[-1].role
    match last_role:
        case "user":
            # The last message is from the user, ask the assistant to respond with a new message
            messages.append(
                Message(role="assistant", parts=[], add_im_end=False, modality="voice")
            )
        case "raw":
            # The last message is raw text, ask the assistant to complete it
            messages[-1].add_im_start = False
            messages[-1].add_im_end = False
            messages[-1].modality = "voice"
        case "assistant":
            # The last message is from the assistant, ask the assistant to continue
            messages[-1].add_im_end = False
        case _:
            # We expect it to be assistant if not user or raw
            raise ValueError("The last message must be from the assistant, user or raw")

    # Create a conversation object and encode it for inference
    conv = Conversation(messages=messages)
    prompt = conv.encode_for_inference(
        tokenizer=tokenizer, num_codebooks=config.num_codebooks
    )
    
    # Use the safe wrapper to get the token ID
    try:
        im_end_id = get_token_id_safe(tokenizer, IM_END_TOKEN)
    except ValueError as e:
        # Fallback: Use -1 as a marker that we couldn't find the token
        # The model will still generate until other end conditions are met
        im_end_id = -1
        print(f"Warning: {str(e)} - using fallback end token detection")

    return prompt, im_end_id


def create_generation_request(prompt, request, im_end_id, device):
    """
    Convert the request into a dictionary that can be sent to the model for generation.
    """
    req = {
        "prompt": prompt.to(device),
        "max_new_tokens": request.max_new_tokens,
        "im_end_id": im_end_id,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "num_samples": request.num_samples,
        "early_stop_threshold": request.early_stop_threshold,
    }
    return req


def send_generation_request(input_queue, req):
    """
    Send the generation request to the model and return a queue to get the response.
    """
    response_queue = queue.Queue()
    input_queue.put(GenerateRequest(req, response_queue))
    return response_queue
