import cv2
from scipy.io.wavfile import write
import sounddevice as sd
import logging
from threading import Thread
import time
import vtii
import requests
import os 
import dotenv

from stttmp import get_text_from_speech

dotenv.load_dotenv()

INTERVAL = 10
URL = os.getenv("SERVER_URI", "https://desktop-dtohfqr.taile61ba3.ts.net")

state_dict = {
    "frames" : []
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
logger.info("Logging setup complete!")


def record_video(state_dict):
    start = time.time()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        state_dict["frames"].append(frame)
        if time.time() - start > INTERVAL:
            break


def record_audio(state_dict):
    # start microphone
    fs = 44100  # Sample rate
    channels = 1
    filename = "tmp.wav"
    myrecording = sd.rec(int(INTERVAL * fs), samplerate=fs, channels=channels)
    sd.wait()

    # stop microphone
    write(filename, fs, myrecording)

def main():
    t1 = Thread(target=record_video, args=(state_dict,))
    t2 = Thread(target=record_audio, args=(state_dict,))
    threads = [t1, t2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    transcript = get_text_from_speech()
    logger.info(f"transcript: {transcript}")
    action = vtii.get_action_from_frames_and_transcript(state_dict["frames"], transcript)
    logger.info(f"action: {action}")
    resp = requests.post(f"{URL}/api/action", json={"content": action})
    logger.info(f"HTTP resp: {resp}")

if __name__ == "__main__":
    main()