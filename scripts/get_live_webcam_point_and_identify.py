import cv2
import numpy as np

from openai import OpenAI

from vtii import point_and_identify
from threading import Thread
import logging
import time
import requests
import queue
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import os

import sounddevice as sd
from scipy.io.wavfile import write
from stttmp import get_text_from_speech


load_dotenv()

URL = os.getenv("SERVER_URI", "https://desktop-dtohfqr.taile61ba3.ts.net")

SINGLE_TASK = False
INTERVAL_SECONDS = 10

state_dict = {"f1_count": 0, "f2_count": 0, "f3_count": 0, "f4_count": 0}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
logger.info("Logging setup complete!")

api_executor = ThreadPoolExecutor(max_workers=10)  # Limit concurrent API calls
results_queue = queue.Queue()


def submit_api_call(frames, idx):
    """Submit API call asynchronously"""

    def worker():
        try:
            result: str = point_and_identify(user_frames=frames)
            payload = {"content": result, "idx": idx}
            r = requests.post(f"{URL}/api/point", json=payload)
            results_queue.put(("success", result))
        except Exception as e:
            results_queue.put(("error", str(e)))

    future = api_executor.submit(worker)
    return future


def f1(main_frames, state_dict):
    logger.info("Starting webcam thread...")
    cap = cv2.VideoCapture(os.getenv("CV2_CAP", 0))
    while True:
        ret, frame = cap.read()
        main_frames.append(frame)

        if SINGLE_TASK:
            if (
                state_dict["f2_count"] > 0
                and state_dict["f3_count"] > 0
                and state_dict["f4_count"] > 0
            ):
                break


def f2(main_frames, interval_seconds, state_dict):
    last_processed_time = time.time()

    while True:
        current_time = time.time()

        # Check for completed API calls
        while not results_queue.empty():
            status, result = results_queue.get()
            if status == "success":
                logger.info(f"Detection result: {result}")
            else:
                logger.error(f"API error: {result}")

        # Submit new API call if needed
        if (
            current_time - last_processed_time >= interval_seconds
            and len(main_frames) > 0
        ):
            logger.info(f">>>>> f2_count: {str(state_dict)}")

            subframes = main_frames.copy()
            main_frames.clear()
            if SINGLE_TASK:
                state_dict["f2_count"] += 1

                logger.info(f"f2: about to point and detect")
                description = point_and_identify(user_frames=subframes)
                logger.info(f"f2: detected a {description}")
                payload = {"content": description, "idx": state_dict["f2_count"] - 1}
                r = requests.post(f"{URL}/api/point", json=payload)
                break

            if subframes:
                logger.info(f"Submitting {len(subframes)} frames for processing")
                submit_api_call(subframes, state_dict["f2_count"])
                logger.info("completed function call for point and detect")
                last_processed_time = current_time

            state_dict["f2_count"] += 1

        time.sleep(0.1)  # TODO: delete?


def f3(state_dict):
    while True:
        fs = 44100  # Sample rate
        channels = 1
        filename = "tmp.wav"
        # audio = sd.rec(int(interval_seconds * fs), samplerate=fs, channels=channels)
        # sd.wait()
        # write(filename, fs, audio)

        recording = []

        def callback(indata, frames, time, status):
            recording.append(indata.copy())

        # Start streaming audio
        with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
            while True:
                if state_dict["f3_count"] < state_dict["f2_count"]:
                    break

        logger.info("Stopped recording.")

        # Combine all recorded chunks
        audio_np = np.concatenate(recording, axis=0)

        # Save to WAV
        write(filename, fs, audio_np)
        logger.info(f">>>>> f3_count: {str(state_dict)}")

        state_dict["f3_count"] += 1
        if SINGLE_TASK:
            break


def f4(state_dict):
    """
    whisper function
    """
    client = OpenAI(api_key=os.getenv("X_OPENAI_API_KEY"))
    while True:
        if state_dict["f4_count"] < state_dict["f3_count"]:
            transcription = get_text_from_speech()
            logger.info(f">>>>>>>>{transcription}")
            # audio_file = open("tmp.wav", "rb")
            # transcription = client.audio.transcriptions.create(
            #     model="whisper-1", file=audio_file
            # )
            # logger.info(f"audio transcription: {transcription}")
            logger.info(f">>>>> f4_count: {str(state_dict)}")

            requests.post(
                os.getenv("SERVER_URI") + "/api/tts",
                json={"content": transcription, "idx": state_dict["f4_count"]},
            )

            state_dict["f4_count"] += 1
            if SINGLE_TASK:
                break


if __name__ == "__main__":
    logging.warning(f"SINGLE_TASK={SINGLE_TASK}")
    main_frames = []
    t_run_webcam = Thread(
        target=f1,
        args=(
            main_frames,
            state_dict,
        ),
    )
    t_run_detection = Thread(
        target=f2,
        args=(
            main_frames,
            INTERVAL_SECONDS,
            state_dict,
        ),
    )
    t_run_mic = Thread(
        target=f3,
        args=(state_dict,),
    )
    t_run_whisper = Thread(target=f4, args=(state_dict,))

    # Make threads daemon so they don't block program exit
    t_run_webcam.daemon = True
    t_run_detection.daemon = True
    t_run_mic.daemon = True
    t_run_whisper.daemon = True

    threads = [t_run_webcam, t_run_detection, t_run_mic, t_run_whisper]
    for t in threads:
        t.start()

    try:
        # Keep the main thread alive
        for t in threads:
            t.join()

    except KeyboardInterrupt:
        print("Shutting down...")
        # Give threads time to finish current operations
        import time

        time.sleep(1)
