import os
import openai
import logging
import tempfile
from scipy.io.wavfile import write as write_wav
import numpy as np
from dotenv import load_dotenv
import io


load_dotenv()

logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    client = openai.OpenAI(api_key=os.getenv("X_OPENAI_API_KEY"))
    # You can set the API key manually if not using environment variables:
    # client = openai.OpenAI(api_key="YOUR_API_KEY")
except openai.OpenAIError as e:
    logger.error(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.")
    client = None

def audio_to_text(audio_chunks, sample_rate):
    """
    Transcribes a list of audio chunks using the OpenAI Whisper API.
    """
    if not client:
        logger.error("OpenAI client not initialized. Cannot transcribe audio.")
        return None
    if not audio_chunks:
        logger.warning("No audio chunks provided for transcription.")
        return ""

    try:
        concatenated_audio = np.concatenate(audio_chunks, axis=0)
        
        # Create in-memory WAV file
        buffer = io.BytesIO()
        write_wav(buffer, sample_rate, concatenated_audio)
        buffer.seek(0)
        
        logger.debug("Transcribing audio from in-memory buffer")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", buffer, "audio/wav")  # Tuple format: (filename, file_object, mime_type)
        )
        
        logger.debug(f"Transcription successful: {transcription.text}")
        return transcription.text
    except Exception as e:
        logger.error(f"Error during OpenAI audio transcription: {e}", exc_info=True)
        return None

def image_to_text(image_data):
    """
    Placeholder for converting image data to text using OpenAI's vision models.

    Args:
        image_data: The image data (format to be determined, e.g., bytes, path, PIL Image).

    Returns:
        str: Description of the image, or None if an error occurs.
    """
    if not client:
        logger.error("OpenAI client not initialized. Cannot process image.")
        return None
    logger.info("image_to_text function called (placeholder).")
    # Example (requires appropriate model and API usage for vision):
    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-4-vision-preview", # Or other vision-capable model
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": "What's in this image?"},
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_image_data}"},
    #                     },
    #                 ],
    #             }
    #         ],
    #         max_tokens=300,
    #     )
    #     return response.choices[0].message.content
    # except Exception as e:
    #     logger.error(f"Error during OpenAI image processing: {e}")
    #     return None
    return "Image description placeholder"

def generate_response(prompt, history=None):
    """
    Generates a text response from OpenAI based on a prompt and conversation history.

    Args:
        prompt (str): The user's input/query.
        history (list of dict, optional): Conversation history, e.g., 
                                         [{"role": "user", "content": "Hello"}, 
                                          {"role": "assistant", "content": "Hi there!"}].

    Returns:
        str: The generated response, or None if an error occurs.
    """
    if not client:
        logger.error("OpenAI client not initialized. Cannot generate response.")
        return None
    
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    try:
        logger.debug(f"Generating response for prompt: \"{prompt}\" with history.")
        completion = client.chat.completions.create(
            model="gpt-4.1",  # Or your preferred model
            messages=messages
        )
        response_text = completion.choices[0].message.content
        logger.debug(f"Generated response: \"{response_text}\"")
        return response_text
    except Exception as e:
        logger.error(f"Error during OpenAI text generation: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example Usage (for testing openai_api.py directly)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger.info("Testing OpenAI API functions...")

    # Test audio_to_text (requires a dummy audio file or mock)
    # For a real test, you'd need some audio data.
    # Example: Create a silent 1-second WAV file for testing structure
    # sample_rate_test = 44100
    # duration_test = 1
    # silence = np.zeros(int(sample_rate_test * duration_test), dtype=np.float32)
    # test_audio_chunks = [silence]
    # transcription = audio_to_text(test_audio_chunks, sample_rate_test)
    # if transcription is not None:
    #     logger.info(f"Test audio_to_text transcription: '{transcription}'")
    # else:
    #     logger.error("Test audio_to_text failed.")

    # Test image_to_text (placeholder)
    # vision_description = image_to_text("dummy_image_data")
    # logger.info(f"Test image_to_text (placeholder): '{vision_description}'")

    # Test generate_response
    test_prompt = "Hello, who are you?"
    test_history = [{"role": "system", "content": "You are a helpful assistant."}]
    text_response = generate_response(test_prompt, test_history)
    if text_response:
        logger.info(f"Test generate_response: '{text_response}'")
    else:
        logger.error("Test generate_response failed.")

    test_prompt_2 = "What was the first thing I said?"
    test_history.append({"role": "user", "content": test_prompt})
    test_history.append({"role": "assistant", "content": text_response if text_response else "I am an AI."})
    text_response_2 = generate_response(test_prompt_2, test_history)
    if text_response_2:
        logger.info(f"Test generate_response 2 (with history): '{text_response_2}'")
    else:
        logger.error("Test generate_response 2 failed.")

