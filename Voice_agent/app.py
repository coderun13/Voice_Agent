import asyncio
import websockets
import assemblyai as aai
import os
import json
import logging
import threading
import queue
import uuid
from dotenv import load_dotenv
from typing import Type
from functools import partial
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

# NEW IMPORTS FOR GEMINI FUNCTION CALLING
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Retrieve API keys
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
murf_api_key = os.getenv("MURF_API_KEY")
murf_voice_id = os.getenv("MURF_VOICE_ID", "en-US-davis")  # Changed voice to Davis

if not assemblyai_api_key or not google_api_key or not murf_api_key:
    raise ValueError("One or more API keys not found in .env file!")

logger.info("API keys loaded successfully.")

# Configure Gemini
genai.configure(api_key=google_api_key)

# Weather function
def get_current_weather(location: str) -> str:
    try:
        import requests

        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=10)
        geo_data = geo_response.json()

        if not geo_data.get('results'):
            return f"I couldn't find weather information for '{location}'. Please try a different city name."

        location_data = geo_data['results'][0]
        lat = location_data['latitude']
        lon = location_data['longitude']
        city_name = location_data['name']
        country = location_data.get('country', '')

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
        weather_response = requests.get(weather_url, timeout=10)
        weather_data = weather_response.json()
        current = weather_data.get('current', {})

        temp = current.get('temperature_2m', '?')
        humidity = current.get('relative_humidity_2m', '?')
        wind_speed = current.get('wind_speed_10m', '?')
        weather_code = current.get('weather_code', -1)

        weather_descriptions = {
            0: "clear skies", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle",
            55: "dense drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
            71: "slight snow", 73: "moderate snow", 75: "heavy snow", 95: "thunderstorm"
        }

        condition = weather_descriptions.get(weather_code, "unknown conditions")

        return f"The weather in {city_name}, {country} is currently {temp}°C with {condition}. Humidity is {humidity}% and wind speed is {wind_speed} km/h."

    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return "I'm having trouble accessing weather information right now. Please try again in a moment."

# Weather tool declaration
weather_function = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather information for a specific location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for (e.g., 'Paris', 'New York', 'Tokyo')"
            }
        },
        "required": ["location"]
    }
)

# Create tool with only weather function
weather_tool = Tool(function_declarations=[weather_function])

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[weather_tool]
)

# Murf TTS API configuration
MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

# A global dictionary to store per-session state, including chat history and locks
active_sessions = {}

# Constants for retrying API calls
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Chat history management with Simple Chatbot persona
def initialize_chat_history():
    return [
        {
            "role": "user",
            "parts": [{"text": """You are a helpful AI voice assistant with weather reporting capabilities.

SKILLS:
1. WEATHER REPORTING: Get current weather for any location worldwide

PERSONALITY:
- Friendly, professional, and helpful
- Natural conversational style with a warm tone
- Concise responses for voice interaction - give ONE clear response only
- Reliable and informative
- Enthusiastic about helping with weather information

IMPORTANT: Always provide exactly ONE response. Do not repeat yourself or give multiple variations of the same answer.

Use the weather function when users ask about weather conditions. You can provide detailed weather information for any location worldwide."""}]
        },
        {
            "role": "model",
            "parts": [{"text": "Hello! I'm your AI weather assistant. I can get current weather information for any location worldwide. Just ask me about the weather in any city and I'll provide you with current conditions including temperature, humidity, wind speed, and weather description. What location would you like to know about?"}]
        }
    ]

def save_chat_turn(session_id, user_message, assistant_response):
    """Save a complete conversation turn to chat history"""
    if session_id not in active_sessions:
        return
    
    chat_history = active_sessions[session_id]['chat_history']
    
    # Add user message
    chat_history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })
    
    # Add assistant response
    chat_history.append({
        "role": "model",
        "parts": [{"text": assistant_response}]
    })
    
    # Keep only last 20 messages to prevent context overflow
    if len(chat_history) > 20:
        # Keep first 2 (system messages) and last 18
        active_sessions[session_id]['chat_history'] = chat_history[:2] + chat_history[-18:]
    
    logger.info(f"Chat history saved for session {session_id}. Total messages: {len(chat_history)}")

async def send_to_murf_websocket(text, session_id):
    """
    Enhanced function to handle Murf WebSocket connection and TTS processing
    Streams individual audio chunks to client as they arrive
    """
    try:
        session_info = active_sessions.get(session_id)
        if not session_info or 'context_id' not in session_info:
            logger.error("Session or context ID not found for Murf TTS.")
            return

        context_id = session_info['context_id']
        
        # Construct Murf WebSocket URL with parameters
        murf_ws_url_full = f"{MURF_WS_URL}?api-key={murf_api_key}&sample_rate=44100&channel_type=MONO&format=WAV&context_id={context_id}"
        
        logger.info(f"Connecting to Murf WebSocket...")
        
        # Create the payload for Murf with changed voice
        murf_payload = {
            "text": text,
            "voice_id": murf_voice_id,  # Now uses the configurable voice ID (defaults to davis)
            "end": True
        }
        
        logger.info(f"Sending to Murf: {json.dumps(murf_payload, indent=2)}")
        
        # Connect to Murf WebSocket with timeout
        async with websockets.connect(
            murf_ws_url_full,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            open_timeout=10
        ) as murf_ws:
            logger.info("Connected to Murf WebSocket successfully!")
            
            # Send the text payload
            await murf_ws.send(json.dumps(murf_payload))
            logger.info("Text payload sent to Murf")
            
            chunk_count = 0
            
            # Send TTS start notification to client
            if session_id in active_sessions:
                message_queue = active_sessions[session_id]['message_queue']
                await message_queue.put({
                    "message_type": "TTSStart",
                    "text": text
                })
            
            # Set a timeout for receiving messages
            timeout_duration = 30  # seconds
            
            try:
                async with asyncio.timeout(timeout_duration):
                    async for message in murf_ws:
                        try:
                            murf_response = json.loads(message)
                            
                            # Check for audio data
                            if 'audio' in murf_response:
                                base64_audio = murf_response['audio']
                                chunk_count += 1
                                
                                logger.info(f"Received Murf audio chunk #{chunk_count}")
                                
                                # Stream individual chunk to client immediately
                                if session_id in active_sessions:
                                    message_queue = active_sessions[session_id]['message_queue']
                                    await message_queue.put({
                                        "message_type": "TTSChunk",
                                        "audio_chunk": base64_audio,
                                        "chunk_number": chunk_count,
                                        "chunk_length": len(base64_audio)
                                    })
                            
                            # Check for errors
                            if 'error' in murf_response:
                                logger.error(f"Murf API error: {murf_response['error']}")
                                if session_id in active_sessions:
                                    message_queue = active_sessions[session_id]['message_queue']
                                    await message_queue.put({
                                        "message_type": "TTSError",
                                        "error": murf_response['error']
                                    })
                                break
                            
                            # Check if this is the final chunk
                            if murf_response.get("final") or murf_response.get("end") or murf_response.get("isFinalAudio"):
                                logger.info("Received final audio chunk from Murf")
                                break
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode Murf response: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing Murf response: {e}")
                            raise
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for Murf response after {timeout_duration}s")
            
            # Send TTS complete notification to client
            if session_id in active_sessions:
                message_queue = active_sessions[session_id]['message_queue']
                await message_queue.put({
                    "message_type": "TTSComplete",
                    "total_chunks": chunk_count
                })
            
    except Exception as e:
        logger.error(f"Unexpected error with Murf: {e}")
        if session_id in active_sessions:
            message_queue = active_sessions[session_id]['message_queue']
            await message_queue.put({
                "message_type": "TTSError",
                "error": f"Unexpected error: {e}"
            })

async def on_llm_api_call(session_id, user_prompt):
    """
    Handles the asynchronous LLM API call using Gemini function calling.
    """
    session_info = active_sessions.get(session_id)
    if not session_info:
        logger.error(f"Session with ID {session_id} not found.")
        return

    llm_lock = session_info['llm_lock']
    message_queue = session_info['message_queue']
    chat_history = session_info['chat_history']

    # If busy, wait for the lock
    if llm_lock.locked():
        logger.info(f"LLM busy for session {session_id} — this prompt will run after the current one.")

    async with llm_lock:
        # Double-check session still active
        if session_id not in active_sessions:
            logger.warning(f"Session {session_id} no longer active. Aborting LLM call.")
            return

        await message_queue.put({"message_type": "LLMStart"})
        logger.info(f"AI weather assistant processing started for session {session_id}...")

        # Retry loop for connection issues
        for attempt in range(MAX_RETRIES):
            try:
                if session_id not in active_sessions:
                    logger.warning(f"Session {session_id} no longer active. Aborting LLM call.")
                    return

                # Build the conversation history for Gemini
                conversation_history = []
                for msg in chat_history:
                    conversation_history.append({
                        "role": msg["role"],
                        "parts": msg["parts"]
                    })
                
                # Add the current user message
                conversation_history.append({
                    "role": "user", 
                    "parts": [{"text": user_prompt}]
                })

                # Start a chat session with the model
                chat = model.start_chat(history=conversation_history[:-1])  # Exclude the latest message
                
                # Send the message and get response with function calling enabled
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: chat.send_message(
                        user_prompt,
                        tools=[weather_tool],
                        tool_config={'function_calling_config': {'mode': 'AUTO'}}
                    )
                )
                
                full_response_text = ""
                
                # Check if there are function calls to handle
                has_function_calls = False
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        has_function_calls = True
                        function_call = part.function_call
                        
                        if function_call.name == "get_current_weather":
                            location = function_call.args["location"]
                            logger.info(f"Weather function called for location: {location}")
                            
                            # Execute the function
                            weather_result = get_current_weather(location)
                            
                            # Send the function result back to the model
                            response = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: chat.send_message([
                                    {
                                        "function_response": {
                                            "name": "get_current_weather",
                                            "response": {"weather": weather_result}
                                        }
                                    }
                                ])
                            )
                            break  # Only handle one function call at a time
                
                # Extract the final text response
                if response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            full_response_text += part.text
                
                if not full_response_text or full_response_text.strip() == "":
                    full_response_text = "I'm having trouble processing your request right now. Please try again."
                
                # Clean up any duplicate responses by checking for repeated patterns
                lines = full_response_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    if line.strip() and line.strip() not in [l.strip() for l in cleaned_lines]:
                        cleaned_lines.append(line)
                
                if cleaned_lines:
                    full_response_text = '\n'.join(cleaned_lines)
                
                # Send the response as chunks for better UX
                words = full_response_text.split()
                chunk_size = 10  # words per chunk
                
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    await message_queue.put({
                        "message_type": "LLMChunk",
                        "text": chunk + (" " if i + chunk_size < len(words) else "")
                    })
                    # Small delay between chunks for better streaming effect
                    await asyncio.sleep(0.1)
                
                # Success - break out of retry loop
                break

            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{MAX_RETRIES} failed with error: {e}")
                if attempt + 1 == MAX_RETRIES:
                    full_response_text = "I'm having some technical difficulties right now. Please try again in a moment."
                    await message_queue.put({"message_type": "LLMError", "error": f"Error: {e}"})
                    break
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        # Save turn + notify completion
        if session_id in active_sessions and full_response_text and full_response_text.strip():
            save_chat_turn(session_id, user_prompt, full_response_text)
            await message_queue.put({"message_type": "LLMComplete", "text": full_response_text})

            # Kick off TTS
            try:
                logger.info(f"Starting TTS for session {session_id}")
                await send_to_murf_websocket(full_response_text, session_id)
            except Exception as e:
                logger.error(f"TTS processing failed: {e}")
                await message_queue.put({"message_type": "TTSError", "error": f"TTS error: {e}"})

class BufferedAudioStream:
    """Audio stream that buffers data to ensure proper chunk sizes"""
    def __init__(self, sample_rate=16000, target_duration_ms=100):
        self.sample_rate = sample_rate
        self.target_duration_ms = target_duration_ms
        self.bytes_per_sample = 2
        self.target_chunk_size = int((sample_rate * target_duration_ms * self.bytes_per_sample) / 1000)
        self.buffer = bytearray()
        self.queue = queue.Queue()
        self.running = True
        logger.info(f"Audio stream initialized: {sample_rate}Hz, {target_duration_ms}ms chunks ({self.target_chunk_size} bytes)")
    
    def add_audio_data(self, data):
        if not self.running:
            return
        self.buffer.extend(data)
        while len(self.buffer) >= self.target_chunk_size:
            chunk = bytes(self.buffer[:self.target_chunk_size])
            self.queue.put(chunk)
            del self.buffer[:self.target_chunk_size]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.running:
            raise StopIteration
        try:
            chunk = self.queue.get(timeout=0.5)
            duration_ms = (len(chunk) / self.bytes_per_sample / self.sample_rate) * 1000
            if duration_ms < 50:
                return b'\x00' * self.target_chunk_size
            return chunk
        except queue.Empty:
            return b'\x00' * self.target_chunk_size
    
    def stop(self):
        self.running = False
        logger.info("Audio stream stopped")

async def send_to_client_task(websocket, message_queue):
    """Task to send messages to the client WebSocket"""
    while not websocket.closed:
        try:
            message = await asyncio.wait_for(message_queue.get(), timeout=1.0)
            message_json = json.dumps(message)
            logger.info(f"Sending to client: {message.get('message_type', 'unknown')}")
            await websocket.send(message_json)
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client WebSocket connection closed")
            break
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            break

async def handle_websocket(websocket, path):
    """Handle WebSocket connection from client"""
    logger.info("New client connected")
    session_id = id(websocket)
    message_queue = asyncio.Queue()
    
    llm_task = None
    send_task = None
    
    try:
        # Get the event loop from the main thread
        main_loop = asyncio.get_running_loop()
        
        audio_stream = BufferedAudioStream(sample_rate=16000, target_duration_ms=100)
        
        streaming_client = StreamingClient(
            StreamingClientOptions(
                api_key=assemblyai_api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        
        # Initialize session with chat history
        active_sessions[session_id] = {
            'websocket': websocket,
            'audio_stream': audio_stream,
            'streaming_client': streaming_client,
            'message_queue': message_queue,
            'chat_history': initialize_chat_history(),  # Initialize with system prompt
            'llm_lock': asyncio.Lock(),
            'context_id': str(uuid.uuid4())  # Dynamic Context ID for each session
        }
        
        # Send welcome message to client
        await message_queue.put({
            "message_type": "WelcomeMessage",
            "text": "Hello! I'm your AI weather assistant. Ask me about the weather in any city around the world!"
        })
        
        def on_begin(client: Type[StreamingClient], event: BeginEvent):
            logger.info(f"Session started: {event.id}")
        
        def on_turn(client: Type[StreamingClient], event: TurnEvent, session_id: int, loop: asyncio.AbstractEventLoop):
            nonlocal llm_task
            try:
                if not event.transcript or not event.transcript.strip():
                    return
                
                message = {
                    "message_type": "FinalTranscript" if event.end_of_turn else "PartialTranscript",
                    "text": event.transcript.strip(),
                    "end_of_turn": event.end_of_turn
                }
                
                if session_id in active_sessions:
                    message_queue = active_sessions[session_id]['message_queue']
                    asyncio.run_coroutine_threadsafe(message_queue.put(message), loop)
                else:
                    logger.warning(f"Session {session_id} not found on turn event. Message skipped.")
                    return

                if event.end_of_turn:
                    logger.info(f"FINAL: {event.transcript}")

                    # Do NOT cancel running LLM tasks; let the lock serialize requests.
                    llm_task = asyncio.run_coroutine_threadsafe(
                        on_llm_api_call(session_id, event.transcript.strip()),
                        loop
                    )
                else:
                    logger.info(f"PARTIAL: {event.transcript}")

            except Exception as e:
                logger.error(f"Error in on_turn: {e}")
                import traceback
                traceback.print_exc()
        
        def on_terminated(client: Type[StreamingClient], event: TerminationEvent):
            logger.info(f"Session terminated: {event.audio_duration_seconds:.1f} seconds processed")
        
        def on_error(client: Type[StreamingClient], error: StreamingError):
            logger.error(f"Streaming error: {error}")
            
        streaming_client.on(StreamingEvents.Begin, on_begin)
        streaming_client.on(StreamingEvents.Turn, partial(on_turn, session_id=session_id, loop=main_loop))
        streaming_client.on(StreamingEvents.Termination, on_terminated)
        streaming_client.on(StreamingEvents.Error, on_error)
        
        def start_streaming():
            try:
                logger.info("Connecting to AssemblyAI v3 Streaming...")
                streaming_client.connect(
                    StreamingParameters(
                        sample_rate=16000,
                        format_turns=True,
                    )
                )
                logger.info("Connected to AssemblyAI!")
                streaming_client.stream(audio_stream)
            except Exception as e:
                logger.error(f"Error in streaming thread: {e}")
                import traceback
                traceback.print_exc()
        
        streaming_thread = threading.Thread(target=start_streaming, daemon=True)
        streaming_thread.start()
        
        send_task = asyncio.create_task(send_to_client_task(websocket, message_queue))
        
        async for message in websocket:
            if isinstance(message, bytes):
                if len(message) > 0:
                    audio_stream.add_audio_data(message)
                else:
                    logger.warning("Received empty audio chunk")
            else:
                # Handle text messages from client (e.g., control commands)
                try:
                    client_message = json.loads(message)
                    if client_message.get("type") == "get_history":
                        # Send chat history to client
                        if session_id in active_sessions:
                            history = active_sessions[session_id]['chat_history']
                            await message_queue.put({
                                "message_type": "ChatHistory",
                                "history": history
                            })
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON text message: {message}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session_id in active_sessions:
            session_info = active_sessions[session_id]
            
            if send_task and not send_task.done():
                send_task.cancel()
            
            if llm_task and not llm_task.done():
                llm_task.cancel()
                try:
                    await asyncio.wait_for(llm_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            
            if 'audio_stream' in session_info:
                session_info['audio_stream'].stop()
            if 'streaming_client' in session_info:
                try:
                    session_info['streaming_client'].disconnect(terminate=True)
                    logger.info("Streaming client disconnected")
                except Exception as close_error:
                    logger.error(f"Error closing streaming client: {close_error}")
            
            del active_sessions[session_id]
            logger.info(f"Session {session_id} cleaned up")

async def main():
    """Start the WebSocket server"""
    logger.info("Starting AI Weather Assistant WebSocket server on ws://localhost:8765")
    
    server = await websockets.serve(handle_websocket, "localhost", 8765)
    logger.info("AI Weather Assistant Server running! Open client.html in your browser.")
    logger.info("Features: Real-time transcription → Gemini weather function → Voice response")
    logger.info("Ask about weather anywhere in the world!")
    logger.info("Press Ctrl+C to stop...")
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nAI Weather Assistant Server stopped!")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        import traceback
        traceback.print_exc()