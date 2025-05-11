import cv2
import threading
import collections
import time
import logging

class VideoCaptureBuffer:
    """
    Continuously captures video frames from a camera or video file and stores them in a thread-safe buffer.
    """
    def __init__(self, source=0, fps=30, buffer_seconds=5):
        self.source = source
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.buffer_maxlen = int(fps * buffer_seconds)
        self.frame_buffer = collections.deque(maxlen=self.buffer_maxlen)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, name="VideoCaptureBufferThread")
        self.thread.daemon = True
        self.logger = logging.getLogger(__name__)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def get_frames(self):
        with self.lock:
            return list(self.frame_buffer)

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video capture device: {self.source}")
            return
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        frame_time_interval = 1.0 / self.fps if self.fps > 0 else 0.04
        while not self.stop_event.is_set():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to grab frame from video capture.")
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame_buffer.append(frame)
            elapsed = time.time() - loop_start
            sleep_time = frame_time_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        cap.release()
        self.logger.info("Video capture stopped.")
