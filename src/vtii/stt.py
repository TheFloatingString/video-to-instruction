from openai import OpenAI
import os
from dotenv import load_dotenv

import sounddevice as sd
from scipy.io.wavfile import write


load_dotenv()

client = OpenAI(api_key=os.getenv("X_OPENAI_API_KEY"))

VERBOSE = True

def get_text_from_speech(duration:int=5):
    '''
    # Settings
    fs = 44100  # Sample rate
    duration = 5  # Duration in seconds
    filename = "tmp.wav"

    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    print("Done recording.")

    # Save as WAV file
    write(filename, fs, audio)
    print(f"Saved to {filename}")
    '''

    audio_file= open("tmp.wav", "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )

    if VERBOSE:
        print(transcription.text)

    return transcription.text

if __name__ == "__main__":
    get_text_from_speech()
