import asyncio
import json
import os
import base64
import sounddevice as sd
import numpy as np
import websockets
import queue
import threading
import datetime
import time
from dotenv import load_dotenv
import cv2
import threading
from video_buffer import VideoCaptureBuffer
from openai_api import image_to_text
from vtii import get_description_for_frame

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("X_OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in .env file")

# Audio settings
SAMPLE_RATE = 24000  # OpenAI Realtime API expects 24kHz
CHUNK_SIZE = 1024
CHANNELS = 1

class VoiceAssistant:
    def __init__(self, system_prompt=None):
        self.websocket = None
        self.recording = False
        self.audio_queue = queue.Queue()  # Thread-safe queue for mic input
        self.playing = False
        self.loop = None
        self.transcript_buffer = ""  # Buffer for complete transcript
        
        # Conversation history
        self.conversation_history = []
        self.current_response = None
        
        # Audio output streaming
        self.audio_output_stream = None
        self.output_thread = None
        self.audio_output_queue = queue.Queue()
        self.output_running = True
        
        # System prompt
        self.system_prompt = system_prompt or "You are a helpful voice assistant."
        
        # Task management
        self.tasks = []
        
        # WebSocket server for web clients
        self.web_clients = set()
        self.web_server = None
        
        # Video buffer for camera frames
        self.video_buffer = None
        self.video_buffer_thread = None
        self.video_buffer_seconds = 5  # How many seconds of frames to keep
        self.video_fps = 30  # Default FPS, can be made configurable
        self.video_source = 0  # Default camera index, can be made configurable
        
        # Microphone blocking flag
        self.microphone_blocked = False
        
    def add_to_history(self, role, content, content_type="text", metadata=None):
        """Add a message to conversation history"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "role": role,
            "content": content,
            "content_type": content_type,
            "metadata": metadata or {}
        }
        self.conversation_history.append(entry)
        
    def get_conversation_history(self, limit=None):
        """Get conversation history, optionally limited to last N entries"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
        
    def print_conversation_history(self):
        """Print the conversation history"""
        print("\n=== Conversation History ===")
        for entry in self.conversation_history:
            timestamp = entry["timestamp"]
            role = entry["role"].upper()
            content = entry["content"]
            print(f"[{timestamp}] {role}: {content}")
        print("===========================\n")
    
    async def handle_web_client(self, websocket):
        """Handle incoming web client connections"""
        # Add client to our set
        self.web_clients.add(websocket)
        print(f"Web client connected. Total clients: {len(self.web_clients)}")
        
        # Send current conversation history to the new client
        await websocket.send(json.dumps({
            "type": "conversation_history",
            "history": self.conversation_history
        }))
        
        try:
            # Wait for client to disconnect
            await websocket.wait_closed()
        finally:
            # Remove client when disconnected
            self.web_clients.remove(websocket)
            print(f"Web client disconnected. Total clients: {len(self.web_clients)}")
    
    async def broadcast_to_web_clients(self, event):
        """Send event to all connected web clients"""
        if self.web_clients:
            # Create a copy of the set to avoid modification during iteration
            disconnected_clients = set()
            
            for client in self.web_clients.copy():
                try:
                    await client.send(json.dumps(event))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    print(f"Error sending to web client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.web_clients.discard(client)
    
    async def start_web_server(self, port=8765):
        """Start the WebSocket server for web clients"""
        # Create a wrapper function that properly handles the method call
        
        self.web_server = await websockets.serve(
            self.handle_web_client, 
            "localhost", 
            port
        )
        print(f"🌐 WebSocket server started on ws://localhost:{port}")
        print("Web clients can connect and receive conversation updates")
        
    async def connect(self):
        """Connect to OpenAI Realtime API"""
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        # Use the older API if available
        try:
            self.websocket = await websockets.connect(url, extra_headers=headers)
        except TypeError:
            # Fallback for newer versions that might use different parameter names
            self.websocket = await websockets.connect(url, additional_headers=headers)
        
        # Configure session for audio input/output with system prompt and function calling
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],  # Enable both text and audio
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 1000
                },
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "alloy",  # Choose voice: alloy, echo, fable, onyx, nova, shimmer
                "temperature": 0.7,
                "instructions": self.system_prompt,  # System prompt
                # ENABLE USER TRANSCRIPTION
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                # UPDATED FUNCTION CALLS
                "tools": [
                    {
                        "type": "function",
                        "name": "pick_up_can",
                        "description": "Control the robot arm to pick up a can of a specified type and place it on the tray.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "can_type": {
                                    "type": "string",
                                    "description": "The type of can to pick up (e.g., 'Coke', 'Pepsi', 'Sprite')."
                                }
                            },
                            "required": ["can_type"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "describe_scene",
                        "description": "Describe what is currently visible in the camera's field of view.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "instructions": {
                                    "type": "string",
                                    "description": "Custom instructions for describing the scene."
                                }
                            },
                            "required": []
                        }
                    },
                    {
                        "type": "function",
                        "name": "identify_pointed_object",
                        "description": "Identify the object that is currently being pointed at in the camera's view.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                ],
                "tool_choice": "auto"
            }
        }
        
        await self.websocket.send(json.dumps(session_config))
        
    async def start_recording(self):
        """Start recording audio from microphone"""
        print("🎤 Listening... (Press Ctrl+C to stop)")
        self.recording = True
        self.loop = asyncio.get_running_loop()
        self.microphone_blocked = False  # Add a flag to block mic

        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            # Only queue audio if not blocked
            if self.recording and not self.microphone_blocked:
                audio_data = indata.copy()
                self.audio_queue.put(audio_data)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.int16,
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            ):
                await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
    
    async def send_audio(self):
        """Send recorded audio to OpenAI"""
        try:
            while True:
                if self.recording:
                    try:
                        # Get audio from thread-safe queue
                        audio_chunk = self.audio_queue.get_nowait()
                        audio_bytes = audio_chunk.tobytes()
                        audio_base64 = base64.b64encode(audio_bytes).decode()
                        
                        audio_message = {
                            "type": "input_audio_buffer.append",
                            "audio": audio_base64
                        }
                        
                        await self.websocket.send(json.dumps(audio_message))
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def block_microphone(self):
        """Block the microphone input (do not queue audio)."""
        self.microphone_blocked = True
        print("🎤 Microphone blocked (agent is speaking)")

    def unblock_microphone(self):
        """Unblock the microphone input (allow audio to be queued)."""
        self.microphone_blocked = False
        print("🎤 Microphone unblocked (agent is listening)")

    def audio_output_worker(self):
        """Worker thread for continuous audio output"""
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
            blocksize=CHUNK_SIZE
        ) as stream:
            self.audio_output_stream = stream
            audio_buffer = b''
            speaking = False
            while self.output_running:
                try:
                    # Get audio data from queue
                    audio_chunk = self.audio_output_queue.get(timeout=0.01)
                    audio_buffer += audio_chunk
                    if not speaking:
                        self.block_microphone()
                        speaking = True
                    # Write when we have enough data
                    while len(audio_buffer) >= CHUNK_SIZE * 2:
                        chunk_to_write = audio_buffer[:CHUNK_SIZE * 2]
                        audio_buffer = audio_buffer[CHUNK_SIZE * 2:]
                        audio_array = np.frombuffer(chunk_to_write, dtype=np.int16)
                        stream.write(audio_array)
                except queue.Empty:
                    # Drain any remaining audio in buffer before unblocking mic
                    while len(audio_buffer) >= CHUNK_SIZE * 2:
                        chunk_to_write = audio_buffer[:CHUNK_SIZE * 2]
                        audio_buffer = audio_buffer[CHUNK_SIZE * 2:]
                        audio_array = np.frombuffer(chunk_to_write, dtype=np.int16)
                        stream.write(audio_array)
                    if len(audio_buffer) > 0:
                        # Write any final partial chunk (pad with zeros if needed)
                        pad_len = CHUNK_SIZE * 2 - len(audio_buffer)
                        chunk_to_write = audio_buffer + b'\x00' * pad_len
                        audio_array = np.frombuffer(chunk_to_write, dtype=np.int16)
                        stream.write(audio_array)
                        audio_buffer = b''
                    if speaking:
                        self.unblock_microphone()
                        speaking = False
                    # Write silence if no data available
                    silence = np.zeros(CHUNK_SIZE, dtype=np.int16)
                    stream.write(silence)
                except Exception as e:
                    if self.output_running:
                        print(f"Audio output error: {e}")
                    break
            # Ensure mic is unblocked on exit
            if speaking:
                self.unblock_microphone()
    
    def start_audio_output(self):
        """Start the audio output thread"""
        self.output_thread = threading.Thread(target=self.audio_output_worker)
        self.output_thread.daemon = True
        self.output_thread.start()
    
    def start_video_buffer(self):
        """Start the video capture buffer for camera frames."""
        if self.video_buffer is None:
            self.video_buffer = VideoCaptureBuffer(
                source=self.video_source,
                fps=self.video_fps,
                buffer_seconds=self.video_buffer_seconds
            )
            self.video_buffer.start()
            print(f"📷 Video buffer started (source={self.video_source}, fps={self.video_fps}, seconds={self.video_buffer_seconds})")

    def stop_video_buffer(self):
        """Stop the video capture buffer."""
        if self.video_buffer:
            self.video_buffer.stop()
            self.video_buffer = None
            print("📷 Video buffer stopped.")
    
    async def handle_function_call(self, function_name, call_id, arguments):
        """Handle function calls from the assistant"""
        try:
            if function_name == "get_current_time":
                result = datetime.datetime.now().strftime("%I:%M %p on %B %d, %Y")
                output = json.dumps({"time": result})
                
            elif function_name == "calculate_expression":
                args = json.loads(arguments)
                expression = args.get("expression", "")
                try:
                    # Safe evaluation of simple mathematical expressions
                    result = eval(expression, {"__builtins__": None}, {})
                    output = json.dumps({"result": result, "expression": expression})
                except Exception as e:
                    output = json.dumps({"error": str(e)})
                    
            elif function_name == "save_note":
                args = json.loads(arguments)
                note = args.get("note", "")
                title = args.get("title", "Note")
                
                # Save to conversation history
                self.add_to_history("user", note, "note", {"title": title})
                output = json.dumps({"status": "saved", "title": title})
                
            elif function_name == "pick_up_can":
                args = json.loads(arguments)
                can_type = args.get("can_type", "")
                output = json.dumps({"status": "success", "can_type": can_type})
                
            elif function_name == "describe_scene":
                # Get the latest frame from the video buffer
                frame = None
                custom_instructions = "Describe the scene in this image as quickly and concisely as possible."
                try:
                    args = json.loads(arguments)
                    if "instructions" in args and args["instructions"].strip():
                        custom_instructions = args["instructions"]
                except Exception:
                    pass
                if self.video_buffer:
                    frames = self.video_buffer.get_frames()
                    if frames:
                        frame = frames[-1]
                if frame is not None:
                    import cv2, base64
                    _, buffer = cv2.imencode('.png', frame)
                    b64_image = base64.b64encode(buffer).decode('utf-8')
                    data_url = f"data:image/png;base64,{b64_image}"
                    description = image_to_text(data_url, instructions=custom_instructions)
                    if not description:
                        description = "Could not get scene description."
                else:
                    description = "No camera frame available."
                output = json.dumps({"description": description})
                
            elif function_name == "identify_pointed_object":
                if self.video_buffer:
                    frames = self.video_buffer.get_frames()
                    if frames:
                        frame = frames[-1]
                        content:str = get_description_for_frame(frame=frame, prompt_mode="single_task", model_name="gpt-4.1")
                        output = json.dumps({"object": f"{content}"})
                
            else:
                output = json.dumps({"error": f"Unknown function: {function_name}"})
            
            # Send function call result back to the API
            function_result = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output
                }
            }
            
            await self.websocket.send(json.dumps(function_result))
            
            # Create a response after function call
            response_create = {
                "type": "response.create"
            }
            
            await self.websocket.send(json.dumps(response_create))
            
        except Exception as e:
            print(f"Error handling function call: {e}")
    
    async def receive_and_handle_events(self):
        """Receive and handle all server events"""
        try:
            while True:
                response = await self.websocket.recv()
                event = json.loads(response)
                
                if event["type"] == "response.audio.delta":
                    # Decode and queue the audio data
                    audio_base64 = event["delta"]
                    audio_bytes = base64.b64decode(audio_base64)
                    self.audio_output_queue.put(audio_bytes)
                    
                elif event["type"] == "response.audio_transcript.delta":
                    # Buffer the transcript parts
                    self.transcript_buffer += event['delta']
                    
                elif event["type"] == "response.audio_transcript.done":
                    # Display the complete transcript and save to history
                    if self.transcript_buffer:
                        print(f"\n🤖: {self.transcript_buffer}")
                        self.add_to_history("assistant", self.transcript_buffer)
                        
                        # Send message to web clients
                        await self.broadcast_to_web_clients({
                            "type": "new_message",
                            "role": "assistant",
                            "content": self.transcript_buffer,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                        self.transcript_buffer = ""  # Reset buffer
                
                elif event["type"] == "input_audio_buffer.speech_started":
                    # User started speaking
                    print("\n🟢 User speaking...")
                    
                elif event["type"] == "input_audio_buffer.speech_stopped":
                    # User stopped speaking
                    print("🔴 User stopped speaking")
                    
                elif event["type"] == "input_audio_buffer.committed":
                    # Audio has been processed and committed
                    print("✅ Audio processed, waiting for transcription...")
                
                # THIS IS THE KEY EVENT FOR USER TRANSCRIPTION
                elif event["type"] == "conversation.item.input_audio_transcription.completed":
                    # User speech transcription is complete
                    transcript = event.get("transcript", "")
                    if transcript:
                        print(f"\n👤: {transcript}")
                        self.add_to_history("user", transcript)
                        
                        # Send message to web clients
                        await self.broadcast_to_web_clients({
                            "type": "new_message",
                            "role": "user",
                            "content": transcript,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                
                elif event["type"] == "response.done":
                    # Check for function calls in the response
                    output = event.get("response", {}).get("output", [])
                    for item in output:
                        if item.get("type") == "function_call":
                            await self.handle_function_call(
                                item.get("name"),
                                item.get("call_id"),
                                item.get("arguments")
                            )
                    
                elif event["type"] == "error":
                    print(f"❌ Error: {event.get('error', {}).get('message', 'Unknown error')}")
                    
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Error receiving event: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        # Stop video buffer
        self.stop_video_buffer()
        
        # Stop audio output
        self.output_running = False
        if self.output_thread:
            self.output_thread.join(timeout=1)
            
        # Cancel all tasks
        self.recording = False
        
        # Close web server
        if self.web_server:
            self.web_server.close()
            await self.web_server.wait_closed()
            
        # Close websocket
        if self.websocket:
            await self.websocket.close()
            
    async def run(self):
        """Main run loop"""
        try:
            await self.connect()
            print("✅ Connected to OpenAI Realtime API")
            print(f"📋 System prompt: {self.system_prompt}")
            print("🔧 Available functions: pick_up_can, describe_scene, identify_pointed_object")
            
            # Start the video buffer for camera frames
            self.start_video_buffer()
            
            # Start the audio output thread
            self.start_audio_output()
            
            # Start the web server
            await self.start_web_server()
            
            # Create tasks
            self.tasks = [
                asyncio.create_task(self.start_recording()),
                asyncio.create_task(self.send_audio()),
                asyncio.create_task(self.receive_and_handle_events())
            ]
            
            # Run tasks concurrently
            await asyncio.gather(*self.tasks)
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            await self.cleanup()

async def main():
    # UPDATED SYSTEM PROMPT
    system_prompt = """You are a voice-controlled assistant for a robot arm with a connected camera. You can:
1. Have natural conversations with the user.
2. Control the robot arm to pick up a can of a specified type (e.g., Le Mate, Redbull, Coke) and place it on the tray.
3. Describe what is currently visible in the camera's field of view.
4. Identify the object that is being pointed at in the camera's view.

When the user asks you to pick up a can, always confirm the type of can before proceeding. Use the describe_scene and identify_pointed_object functions to help answer questions about the environment. Be concise, helpful, and always clarify if you are unsure about the user's intent."""
    
    assistant = VoiceAssistant(system_prompt=system_prompt)
    
    try:
        await assistant.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n👋 Goodbye!")
        print("\n=== Final Conversation History ===")
        assistant.print_conversation_history()

if __name__ == "__main__":
    print("🎙️ Enhanced Voice Assistant with OpenAI Realtime API")
    print("----------------------------------------")
    print("Features:")
    print("  • Conversation history tracking")
    print("  • Custom system prompts")
    print("  • Function calling (pick_up_can, describe_scene, identify_pointed_object)")
    print("  • WebSocket server for web clients")
    print("----------------------------------------")
    print("Make sure you have your OPENAI_API_KEY in a .env file")
    print("Press Ctrl+C to exit and see conversation history")
    print("Web clients can connect to ws://localhost:8765")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")