import cv2
import sounddevice as sd
import numpy as np
import collections
import threading
import time
import logging
import os
import random  # For random sleep in task
from scipy.io.wavfile import write as write_wav  # For saving audio

from agent_thread import AgentThread  # Import the new AgentThread
from openai_api import audio_to_text  # Import the audio transcription function

# Setup logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO for more verbose output

# Configuration
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))  # Frames per second
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "44100"))  # Samples per second
AUDIO_CHUNK_DURATION_SECONDS = float(os.getenv("AUDIO_CHUNK_DURATION_SECONDS", "0.1"))  # Duration of each audio chunk
BUFFER_DURATION_SECONDS = int(os.getenv("BUFFER_DURATION_SECONDS", "5"))  # Duration of the rolling window in seconds
TASK_INTERVAL_SECONDS = 5  # How often to schedule a new task

VIDEO_BUFFER_MAXLEN = VIDEO_FPS * BUFFER_DURATION_SECONDS
AUDIO_CHUNKS_PER_BUFFER = int(BUFFER_DURATION_SECONDS / AUDIO_CHUNK_DURATION_SECONDS)
AUDIO_FRAMES_PER_CHUNK = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION_SECONDS)

# Shared circular buffers and locks
video_buffer = collections.deque(maxlen=VIDEO_BUFFER_MAXLEN)
audio_buffer = collections.deque(maxlen=AUDIO_CHUNKS_PER_BUFFER)  # Stores chunks of audio data

video_lock = threading.Lock()
audio_lock = threading.Lock()

# New synchronization primitives for ordered printing
print_order_lock = threading.Lock()
print_order_condition = threading.Condition(print_order_lock)  # Associated with print_order_lock
last_printed_task_id = -1  # Task IDs start from 0
task_id_counter = 0  # Global counter for task IDs

stop_event = threading.Event()  # Event to signal threads to stop
agent_instance = None  # Global variable to hold the agent instance

def video_capture_thread():
    """
    Continuously captures video frames from the camera and writes to video_buffer.
    """
    logger.info("Video capture thread started.")
    cap_source_env = os.getenv("CV2_CAP")
    if cap_source_env is None:
        cap_source = 0  # Default camera
        logger.info(f"CV2_CAP environment variable not set, using default camera index {cap_source}.")
    elif cap_source_env.isdigit():
        cap_source = int(cap_source_env)
        logger.info(f"Using camera index from CV2_CAP: {cap_source}.")
    else:
        cap_source = cap_source_env  # Path to video file
        logger.info(f"Using video file from CV2_CAP: {cap_source}.")
    
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video capture device: {cap_source}")
        return

    # Attempt to set FPS, though it might not be respected by all cameras/drivers
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.debug(f"Requested FPS: {VIDEO_FPS}, Actual FPS from camera: {actual_fps if actual_fps > 0 else 'N/A'}")

    frame_time_interval = 1.0 / VIDEO_FPS if VIDEO_FPS > 0 else 0.04  # Approx 25 FPS if VIDEO_FPS is 0

    while not stop_event.is_set():
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame from video capture. If this is a file, it might have ended.")
            # If it's a file source and it ended, we can break or wait.
            if isinstance(cap_source, str):  # Check if source is a file path
                logger.info("Video file source ended. Stopping video capture.")
                break
            time.sleep(0.1)
            continue
        
        with video_lock:
            video_buffer.append(frame)
        
        # Control frame rate
        elapsed_time = time.time() - loop_start_time
        sleep_time = frame_time_interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    logger.info("Video capture thread stopped.")

def audio_capture_thread():
    """
    Continuously captures audio input from the microphone and writes to audio_buffer.
    Each item in the buffer is a chunk of audio data (numpy array).
    """
    logger.info("Audio capture thread started.")
    
    def audio_callback(indata, frames, time_info, status):
        """
        This callback is called by sounddevice for each new audio chunk.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        with audio_lock:
            audio_buffer.append(indata.copy())  # Add a copy of the audio data chunk

    try:
        logger.debug(f"Attempting to open audio input stream with SR={AUDIO_SAMPLE_RATE}, Channels=1, Blocksize={AUDIO_FRAMES_PER_CHUNK}")
        # Using InputStream for continuous capture
        with sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,  # Mono audio
            dtype='float32',  # Standard data type
            blocksize=AUDIO_FRAMES_PER_CHUNK,  # Number of frames per callback
            callback=audio_callback
        ):
            while not stop_event.is_set():
                time.sleep(0.1)  # Keep the thread alive and check stop_event periodically
    except Exception as e:
        logger.error(f"Audio capture error: {e}")
        logger.error("Please ensure you have a microphone connected and sounddevice is configured correctly.")
        logger.error("Available audio devices: \n" + str(sd.query_devices()))
    finally:
        logger.info("Audio capture thread stopped.")

def process_task_data(task_id, video_frames_snapshot, audio_chunks_snapshot, agent_ref):
    """
    Processes video and audio data. Audio is transcribed and sent to the Agent.
    The final print statement is synchronized to ensure sequential order.
    """
    global last_printed_task_id

    logger.info(f"Task {task_id}: Starting processing with {len(video_frames_snapshot)} video frames and {len(audio_chunks_snapshot)} audio chunks.")
    
    transcribed_text = None
    if audio_chunks_snapshot:
        logger.debug(f"Task {task_id}: Transcribing {len(audio_chunks_snapshot)} audio chunks using OpenAI API.")
        transcribed_text = audio_to_text(audio_chunks_snapshot, AUDIO_SAMPLE_RATE)
        
        if transcribed_text is not None and transcribed_text.strip() != "":
            logger.debug(f"Task {task_id}: Transcription successful. Text: '{transcribed_text[:60]}...'")
            agent_ref.add_task_data(task_id, "transcribed_audio", transcribed_text)
        elif transcribed_text == "":
            logger.debug(f"Task {task_id}: Transcription resulted in empty text (likely silence).")
        else:  # None was returned, indicating an error during transcription
            logger.error(f"Task {task_id}: Audio transcription failed or returned None.")
    else:
        logger.debug(f"Task {task_id}: No audio chunks to process.")
    
    # Synchronize the final print statement for task completion order
    with print_order_condition:
        while last_printed_task_id != task_id - 1:
            print_order_condition.wait(timeout=0.5)  # Wait with a timeout
            if stop_event.is_set():  # Check if shutdown initiated
                logger.info(f"Task {task_id}: Stop event detected during print wait, exiting.")
                return  # Exit if stop event is set
        
        status_message = "Data sent to agent" if transcribed_text and transcribed_text.strip() else "No new audio data for agent"
        logger.info(f"Task {task_id}: Processing finished. Status: {status_message}.")
        last_printed_task_id = task_id
        print_order_condition.notify_all()

def task_scheduling_thread(agent_ref):
    """
    Periodically extracts data from buffers and spawns new task processing threads.
    Passes a reference to the agent to each task.
    """
    global task_id_counter
    logger.info("Task scheduling thread started.")

    while not stop_event.is_set():
        # Wait for the defined interval before scheduling the next task
        # Check stop_event frequently to allow quick shutdown
        for _ in range(TASK_INTERVAL_SECONDS * 10):  # Check every 0.1s
            if stop_event.is_set():
                break
            time.sleep(0.1)
        if stop_event.is_set():
            break

        logger.debug("Scheduler: Time to schedule a new task.")
        
        current_video_frames = []
        current_audio_chunks = []

        # Safely get a snapshot of the current buffers
        with video_lock:
            if video_buffer:  # Check if buffer is not empty
                current_video_frames = list(video_buffer)  # Create a copy
        
        with audio_lock:
            if audio_buffer:  # Check if buffer is not empty
                current_audio_chunks = list(audio_buffer)  # Create a copy

        if not current_video_frames and not current_audio_chunks:
            logger.debug("Scheduler: Buffers are empty, skipping task creation this cycle.")
            continue

        task_id = task_id_counter
        task_id_counter += 1
        
        logger.debug(f"Scheduler: Spawning Task {task_id}.")
        # Create a new thread for the task.
        task_thread = threading.Thread(
            target=process_task_data,
            args=(task_id, current_video_frames, current_audio_chunks, agent_ref),
            name=f"TaskProcessingThread-{task_id}"
        )
        task_thread.daemon = True  # Allow main program to exit even if tasks are queued
        task_thread.start()

    logger.info("Task scheduling thread stopped.")

if __name__ == "__main__":
    logger.info("Starting application...")
    # Ensure OPENAI_API_KEY is set for the application to function fully
    if not os.getenv("X_OPENAI_API_KEY"):
        logger.warning("X_OPENAI_API_KEY environment variable is not set. OpenAI API calls will fail.")

    logger.info(f"Video Buffer Max Length: {VIDEO_BUFFER_MAXLEN} frames ({BUFFER_DURATION_SECONDS}s @ {VIDEO_FPS} FPS)")
    logger.info(f"Audio Buffer Max Length: {AUDIO_CHUNKS_PER_BUFFER} chunks ({BUFFER_DURATION_SECONDS}s, each chunk {AUDIO_CHUNK_DURATION_SECONDS}s)")

    # Create and start the Agent Thread
    agent_instance = AgentThread()
    agent_instance.start()

    # Create and start other threads
    video_thread = threading.Thread(target=video_capture_thread, name="VideoCaptureThread")
    audio_thread = threading.Thread(target=audio_capture_thread, name="AudioCaptureThread")
    scheduler_thread = threading.Thread(target=task_scheduling_thread, args=(agent_instance,), name="TaskSchedulingThread")

    video_thread.daemon = True  # Allow main program to exit even if threads are running
    audio_thread.daemon = True
    scheduler_thread.daemon = True

    video_thread.start()
    audio_thread.start()
    scheduler_thread.start()

    try:
        # Keep the main thread alive to monitor (or for other tasks)
        # For demonstration, let's print buffer sizes periodically
        while True:
            time.sleep(2)
            with video_lock:
                video_len = len(video_buffer)
            with audio_lock:
                audio_len = len(audio_buffer)
            logger.debug(f"Video buffer size: {video_len}/{VIDEO_BUFFER_MAXLEN} frames")
            logger.debug(f"Audio buffer size: {audio_len}/{AUDIO_CHUNKS_PER_BUFFER} chunks")
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping threads...")
        stop_event.set()  # Signal all threads to stop their loops

        logger.info("Stopping agent thread...")
        if agent_instance:
            agent_instance.stop()
            # Wait for agent thread to finish its current work and exit run loop
            if agent_instance.is_alive():
                agent_instance.join(timeout=5) 
            if agent_instance.is_alive():
                logger.warning("Agent thread did not stop in time.")
            else:
                logger.info("Agent thread stopped successfully.")

        # Other threads (video, audio, scheduler) will see stop_event and exit their loops.
        # Give them a moment before trying to join.
        # The task processing threads are daemon and will be handled by print_order_condition timeout or stop_event check.

        # Save buffers before exiting
        logger.info("Saving video and audio buffers...")
        
        # Save video buffer as a video file
        with video_lock:
            if video_buffer:
                logger.info(f"Saving {len(video_buffer)} video frames as a video file.")
                output_video_path = "output_video.mp4"  # Or .avi, depending on desired codec
                
                # Get frame dimensions from the first frame
                first_frame = video_buffer[0]
                height, width, layers = first_frame.shape
                
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                
                effective_fps = VIDEO_FPS if VIDEO_FPS > 0 else 30  # Default to 30 if not set or 0
                
                out_video = cv2.VideoWriter(output_video_path, fourcc, effective_fps, (width, height))

                for frame in list(video_buffer):  # Iterate over a copy
                    out_video.write(frame)
                
                out_video.release()
                logger.info(f"Video buffer saved to {output_video_path}")
            else:
                logger.info("Video buffer is empty. Nothing to save.")

        # Save audio buffer
        with audio_lock:
            if audio_buffer:
                logger.info(f"Saving {len(audio_buffer)} audio chunks.")
                concatenated_audio = np.concatenate(list(audio_buffer))  # Iterate over a copy
                output_audio_path = "output_audio.wav"
                try:
                    write_wav(output_audio_path, AUDIO_SAMPLE_RATE, concatenated_audio)
                    logger.info(f"Audio buffer saved to {output_audio_path}")
                except Exception as e:
                    logger.error(f"Failed to save audio buffer: {e}")
            else:
                logger.info("Audio buffer is empty. Nothing to save.")

        # Wait for threads to finish
        if video_thread.is_alive():
            logger.info("Waiting for video thread to stop...")
            video_thread.join(timeout=5)  # Wait up to 5 seconds
        if audio_thread.is_alive():
            logger.info("Waiting for audio thread to stop...")
            audio_thread.join(timeout=5)  # Wait up to 5 seconds
        if scheduler_thread.is_alive():
            logger.info("Waiting for scheduler thread to stop...")
            scheduler_thread.join(timeout=TASK_INTERVAL_SECONDS + 2)

        logger.info("Allowing a moment for active data capture and task threads to complete...")
        time.sleep(1)  # Brief pause for threads to react to stop_event

        if video_thread.is_alive():
            logger.warning("Video thread did not stop in time.")
        if audio_thread.is_alive():
            logger.warning("Audio thread did not stop in time.")
        if scheduler_thread.is_alive():
            logger.warning("Scheduler thread did not stop in time.")
            
        logger.info("All threads processed for shutdown. Exiting application.")

