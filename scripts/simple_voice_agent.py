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
                # Function calling configuration
                "tools": [
                    {
                        "type": "function",
                        "name": "get_current_time",
                        "description": "Get the current time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "type": "function",
                        "name": "calculate_expression",
                        "description": "Calculate a mathematical expression",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "The mathematical expression to calculate (e.g., '2+2' or '10*5')"
                                }
                            },
                            "required": ["expression"]
                        }
                    },
                    {
                        "type": "function",
                        "name": "save_note",
                        "description": "Save a note to the conversation history",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "note": {
                                    "type": "string",
                                    "description": "The note content to save"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Optional title for the note"
                                }
                            },
                            "required": ["note"]
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
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            if self.recording:
                audio_data = indata.copy()
                # Put audio data in thread-safe queue
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
            while self.output_running:
                try:
                    # Get audio data from queue
                    audio_chunk = self.audio_output_queue.get(timeout=0.01)
                    audio_buffer += audio_chunk
                    
                    # Write when we have enough data
                    while len(audio_buffer) >= CHUNK_SIZE * 2:  # 2 bytes per int16 sample
                        chunk_to_write = audio_buffer[:CHUNK_SIZE * 2]
                        audio_buffer = audio_buffer[CHUNK_SIZE * 2:]
                        
                        # Convert bytes to numpy array
                        audio_array = np.frombuffer(chunk_to_write, dtype=np.int16)
                        
                        # Write to the stream
                        stream.write(audio_array)
                        
                except queue.Empty:
                    # Write silence if no data available
                    silence = np.zeros(CHUNK_SIZE, dtype=np.int16)
                    stream.write(silence)
                except Exception as e:
                    if self.output_running:
                        print(f"Audio output error: {e}")
                    break
    
    def start_audio_output(self):
        """Start the audio output thread"""
        self.output_thread = threading.Thread(target=self.audio_output_worker)
        self.output_thread.daemon = True
        self.output_thread.start()
    
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
                        buffer = self.transcript_buffer
                        print(f"🤖: {buffer}")
                        self.add_to_history("assistant", buffer)
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
                    transcript = event.get("transcript", "")[:-1]
                    if transcript:
                        print(f"\n👤: {transcript}")
                        self.add_to_history("user", transcript)
                
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
        
        # Stop audio output
        self.output_running = False
        if self.output_thread:
            self.output_thread.join(timeout=1)
            
        # Cancel all tasks
        self.recording = False
        
        # Close websocket
        if self.websocket:
            await self.websocket.close()
            
    async def run(self):
        """Main run loop"""
        try:
            await self.connect()
            print("✅ Connected to OpenAI Realtime API")
            print(f"📋 System prompt: {self.system_prompt}")
            print("🔧 Available functions: get_current_time, calculate_expression, save_note")
            
            # Start the audio output thread
            self.start_audio_output()
            
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
    # You can customize the system prompt here
    system_prompt = """You are a helpful voice assistant. You can:
    1. Have natural conversations
    2. Tell the current time
    3. Do mathematical calculations
    4. Save notes for later
    
    Be concise and friendly in your responses."""
    
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
    print("  • Function calling (time, calculator, notes)")
    print("-------------------------------- --------")
    print("Make sure you have your OPENAI_API_KEY in a .env file")
    print("Press Ctrl+C to exit and see conversation history")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")