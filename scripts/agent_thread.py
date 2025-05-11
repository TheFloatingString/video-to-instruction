import threading
import queue
import logging
import time
import os
from openai_api import generate_response # Import from the new openai_api.py

logger = logging.getLogger(__name__)

class AgentThread(threading.Thread): # Renamed and inherits from threading.Thread
    def __init__(self):
        super().__init__() # Call to superclass constructor
        self.input_queue = queue.Queue()
        self.conversation_history = []
        self.running = True # Controls the main loop
        self.name = "AgentThread" # Set thread name for logging
        self.daemon = True

    def make_decision(self, input_text):
        """
        Processes the input text, generates a response using OpenAI, and updates history.
        """
        logger.info(f"Agent received: '{input_text}'")
        
        # Generate response using OpenAI API
        # Pass current conversation history to maintain context
        agent_response_text = generate_response(input_text, self.conversation_history)

        if agent_response_text is None:
            agent_response_text = "Sorry, I encountered an error trying to generate a response."
            logger.error("Agent failed to generate a response from OpenAI.")
        
        self.update_history(input_text, agent_response_text)
        logger.info(f"Agent response: '{agent_response_text}'")
        # logger.debug(f"Current conversation history: {self.conversation_history}")
        return agent_response_text

    def update_history(self, user_input, agent_response):
        """
        Maintains the conversation history.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": agent_response})
        # Optional: Limit history size
        # MAX_HISTORY_LEN = 20 # e.g., keep last 10 pairs
        # if len(self.conversation_history) > MAX_HISTORY_LEN:
        #     self.conversation_history = self.conversation_history[-MAX_HISTORY_LEN:]

    def run(self):
        logger.info("Agent thread started. Waiting for tasks...")
        while self.running:
            try:
                # item is expected to be (task_id, data_type, content)
                task_id, data_type, content = self.input_queue.get(timeout=1)
                
                if data_type == "transcribed_audio":
                    logger.debug(f"Agent processing task {task_id} (type: {data_type}).")
                    self.make_decision(content)
                # Add more data_type handlers here if needed in the future
                # elif data_type == "video_analysis_result":
                #     self._process_video_data(content) 
                else:
                    logger.warning(f"Agent received unknown data type from task {task_id}: {data_type}")

                self.input_queue.task_done()
            except queue.Empty:
                continue # Continue loop to check self.running
            except Exception as e:
                logger.error(f"Error in agent thread's run loop: {e}", exc_info=True)
        logger.info("Agent thread run loop ended.")

    def stop(self):
        logger.info("Agent stop signal received.")
        self.running = False
        # self.input_queue.put(None) # Optional: Unblock queue.get if it's waiting indefinitely
        # No need to join here if daemon=True, main thread will handle shutdown wait if necessary

    def add_task_data(self, task_id, data_type, content):
        if self.running:
            self.input_queue.put((task_id, data_type, content))
        else:
            logger.warning("Agent is not running, not adding new task data.")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, # Use DEBUG for more verbose output during testing
        format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    test_agent = AgentThread()
    test_agent.start()

    try:
        logger.info("Test: Simulating tasks sending data to agent.")
        # Ensure your OPENAI_API_KEY is set in your environment for this test to work
        if os.getenv("OPENAI_API_KEY"):
            test_agent.add_task_data(0, "transcribed_audio", "Hello agent, what is the weather like today?")
            time.sleep(5) # Give agent time to process and call API
            test_agent.add_task_data(1, "transcribed_audio", "And what about tomorrow?")
            time.sleep(5) 
            test_agent.add_task_data(2, "unknown_data", "This is some other data.")
            time.sleep(2)
        else:
            logger.warning("OPENAI_API_KEY not set. Skipping API call tests for agent.")
            test_agent.add_task_data(0, "transcribed_audio", "Test input without API.")
            time.sleep(2)


    except KeyboardInterrupt:
        logger.info("Test script interrupted by user.")
    finally:
        logger.info("Test: Stopping agent.")
        test_agent.stop()
        if test_agent.is_alive():
            test_agent.join(timeout=5)
        logger.info("Test script finished.")
