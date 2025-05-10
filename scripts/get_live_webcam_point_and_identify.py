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


# class Webcam:
#     def __init__(self, width=640, height=480, fps=30, device_index=0):
#         self.cap = cv2.VideoCapture(device_index)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         self.cap.set(cv2.CAP_PROP_FPS, fps)
#         self.display_text = "ack"
#         self.frame_q = []

#         if not self.cap.isOpened():
#             raise RuntimeError("Could not open webcam")

#     def update_display_text(self, updated_text):
#         self.display_text = updated_text

#     def get_frame(self):
#         ret, frame = self.cap.read()
#         self.frame_q.append(frame)
#         frame = cv2.putText(
#             frame,
#             self.display_text,
#             (50, 50),
#             cv2.FONT_HERSHEY_COMPLEX,
#             1,
#             (0, 255, 0),
#             2,
#         )
#         if not ret or frame.size == 0:
#             return None

#         return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     def stop(self):
#         if self.cap.isOpened():
#             self.cap.release()

#     def dump_and_delete_frames(self):
#         return_frames = self.frame_q.copy()
#         self.frame_q = []
#         return return_frames

#     def get_frames_length(self):
#         return len(self.frame_q)


# def run_webcam(cam: Webcam) -> None:
#     frames = []

#     while True:
#         logging.info(f"camera is reading.")
#         rgb_frame = cam.get_frame()
#         logging.info(f"{rgb_frame.shape}")

#         if rgb_frame is not None:
#             cv2.imshow("Webcam Output", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
#             frames.append(rgb_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

        # if len(frames) > 100:
        # break

    # frames = cam.dump_and_delete_frames()
    # # logger.info("processing pointed content")

    # content = point_and_identify(user_frames=frames)
    # print(len(content))
    # print(content)
    # # logger.info(content)


# def run_detection(cam: Webcam) -> None:
#     while True:
#         logger.info(f"fl: {cam.get_frames_length()}")
#         if cam.get_frames_length() > 100:
#             frames = cam.dump_and_delete_frames()
#             logger.info("processing pointed content")
#             content = point_and_identify(user_frames=frames)
#             print(len(content))
#             logger.info(content)
#             # break
#             cam.update_display_text(updated_text=content)
#             # break


def f1(main_frames):
    cap = cv2.VideoCapture(0)
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

#     # finally:
#     #     cam.stop()
#     #     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     cam = Webcam()
#     print("ack")
#     list_of_frames = []
#     while True:
#         frame = cam.get_frame()
#         # list_of_frames.append(frame)
#         if cam.get_frames_length()>100:
#             break
#             print("break")
    
#     content = point_and_identify(user_frames=cam.dump_and_delete_frames())
#     print(len(content))
#     print(content)
