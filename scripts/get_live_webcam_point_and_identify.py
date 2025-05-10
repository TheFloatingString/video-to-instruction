import cv2
import numpy as np

from vtii import point_and_identify
from threading import Thread
import logging

import requests

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("application.log")],
)

logger = logging.getLogger("Main")
logger.info("Logging setup complete!")


def f1(main_frames):
    cap = cv2.VideoCapture(1)
    while True:
        logging.info("f1")
        ret, frame = cap.read()
        logging.info(f"f1: {frame.shape}")
        main_frames.append(frame)


def f2(main_frames):
    while True:
        logging.info(f"f2: {len(main_frames)}")
        if len(main_frames) > 100:
            subframes = main_frames.copy()
            main_frames.clear()
            content = None
            # x = requests.get("https://www.google.com")
            content = point_and_identify(user_frames=subframes)
            logger.info(content)




if __name__ == "__main__":
    # cam = Webcam()
    main_frames = []
    t_run_webcam = Thread(target=f1, args=(main_frames,))
    t_run_detection = Thread(target=f2, args=(main_frames,))
    threads = [t_run_webcam, t_run_detection]
    for t in threads:
        t.start()
