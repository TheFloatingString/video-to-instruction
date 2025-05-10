import cv2
import numpy as np

from vtii import point_and_identify
from threading import Thread
import logging
import time
import requests
import queue
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

URL = "https://desktop-dtohfqr.taile61ba3.ts.net"

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
logger.info("Logging setup complete!")

api_executor = ThreadPoolExecutor(max_workers=5)  # Limit concurrent API calls
results_queue = queue.Queue()

def submit_api_call(frames):
    """Submit API call asynchronously"""
    def worker():
        try:
            result: str = point_and_identify(user_frames=frames)
            payload = {"content": result}
            r = requests.post(f"{URL}/api/point", json=payload)
            results_queue.put(("success", result))
        except Exception as e:
            results_queue.put(("error", str(e)))
    
    future = api_executor.submit(worker)
    return future

def f1(main_frames):
    logger.info("Starting webcam thread...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        main_frames.append(frame)


def f2(main_frames, interval_seconds=5.0):
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
        if current_time - last_processed_time >= interval_seconds and len(main_frames) > 0:
            subframes = main_frames.copy()
            main_frames.clear()
            
            if subframes:
                logger.info(f"Submitting {len(subframes)} frames for processing")
                submit_api_call(subframes)
                last_processed_time = current_time
        
        time.sleep(0.1)


if __name__ == "__main__":
    main_frames = []
    t_run_webcam = Thread(target=f1, args=(main_frames,))
    t_run_detection = Thread(target=f2, args=(main_frames,))
    
    # Make threads daemon so they don't block program exit
    t_run_webcam.daemon = True
    t_run_detection.daemon = True
    
    threads = [t_run_webcam, t_run_detection]
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
