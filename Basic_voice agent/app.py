from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import assemblyai as aai
from uuid import uuid4

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Mount static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Uploads directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API keys
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Initialize transcriber
transcriber = aai.Transcriber()

# Using a simple in-memory dictionary as the datastore for chat history
# Format: { session_id: [ {"role": "user"/"assistant", "content": "..."} ] }
chat_history = {}

# ---------- DAY 1: Serve UI ----------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- DAY 2: Voice Generation ----------
class TTSRequest(BaseModel):
    text: str
    voice_id: str

@app.post("/generate")
def generate_voice(req: TTSRequest):
    headers = {
        "Authorization": f"Bearer {MURF_API_KEY}",
        "Content-Type": "application/json",
        "api-key": f"{MURF_API_KEY}"
    }

    body = {
        "voiceId": req.voice_id,
        "text": req.text
    }

    response = requests.post("https://api.murf.ai/v1/speech/generate", json=body, headers=headers)
    if response.status_code == 200:
        return {"audio_url": response.json().get("audioFile")}
    else:
        return {"error": response.text}

# ---------- DAY 5: Upload Audio ----------
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": os.path.getsize(file_location)
    }

# ---------- DAY 6: Transcribe Audio File ----------
@app.post("/transcribe/file")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Transcribe using AssemblyAI
        transcript = transcriber.transcribe(temp_path)

        # Delete the temporary file
        os.remove(temp_path)

        return {"transcription": transcript.text}
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

# ---------- DAY 7: Echo Bot v2 ----------
@app.post("/tts/echo")
async def tts_echo(file: UploadFile = File(...)):
    try:
        # 1. Save uploaded audio temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 2. Transcribe with AssemblyAI
        transcript = transcriber.transcribe(temp_path)

        if not transcript.text or transcript.status != "completed":
            os.remove(temp_path)
            return JSONResponse(status_code=500, content={"error": "Transcription failed"})

        text = transcript.text

        # 3. Send transcription to Murf for TTS
        headers = {
            "Authorization": f"Bearer {MURF_API_KEY}",
            "Content-Type": "application/json",
            "api-key": f"{MURF_API_KEY}"
        }
        body = {
            "voiceId": "en-UK-hazel",  # Can change this to any Murf voice ID
            "text": text
        }
        murf_response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            json=body,
            headers=headers
        )

        # 4. Clean up temp file
        os.remove(temp_path)

        # 5. Return both transcription and audio
        if murf_response.status_code == 200:
            audio_url = murf_response.json().get("audioFile")
            return {
                "transcription": text,
                "audio_url": audio_url
            }
        else:
            return JSONResponse(status_code=500, content={"error": murf_response.text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- DAY 8 & 9: LLM Query (Text + Audio) ----------
@app.post("/llm/query")
async def llm_query(
    request: Request,
    audio: UploadFile = File(None), # Day 9: Audio input for voice pipeline
    file: UploadFile = File(None),  # Backward compatibility
    text: str = Form(None)          # Day 8: Text input + Form data support
):
    """
    Combined Day 8 & 9 functionality:
    - Day 8: Accept text input, send to LLM, return text response
    - Day 9: Accept audio input, transcribe, send to LLM, return voice response
    """
    try:
        # Validate API keys
        if not GEMINI_API_KEY:
            return JSONResponse(status_code=500, content={"error": "GEMINI_API_KEY not set"})

        user_text = ""
        transcription = ""
        is_voice_input = False

        # Check if request is JSON (Day 8 text-only) or FormData (Day 9 voice or mixed)
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # Day 8: Pure JSON text request
            try:
                json_body = await request.json()
                user_text = json_body.get("text", "").strip()
                if not user_text:
                    return JSONResponse(status_code=400, content={"error": "Text is required"})
                print(f" Day 8 Text query: '{user_text[:100]}...'")
            except Exception:
                return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
        
        else:
            # Day 9: FormData request (can have audio and/or text)
            audio_file = audio or file  # Use 'audio' parameter, fallback to 'file'
            
            if audio_file:
                # Audio input - Day 9 pipeline
                is_voice_input = True
                print(f" Day 9 Voice query: {audio_file.filename}")
                
                if not ASSEMBLYAI_API_KEY:
                    return JSONResponse(status_code=500, content={"error": "ASSEMBLYAI_API_KEY not set"})
                
                # Save and transcribe audio
                temp_path = f"temp_{audio_file.filename}"
                
                try:
                    with open(temp_path, "wb") as f:
                        content = await audio_file.read()
                        f.write(content)
                    
                    print(f"🎵 Audio saved: {len(content)} bytes")

                    # Transcribe audio
                    transcript = transcriber.transcribe(temp_path)
                    os.remove(temp_path)

                    if not transcript.text or transcript.status != "completed":
                        return JSONResponse(
                            status_code=500, 
                            content={"error": f"Transcription failed. Status: {transcript.status}"}
                        )

                    user_text = transcript.text.strip()
                    transcription = user_text
                    print(f" Transcribed: '{user_text[:100]}...'")

                except Exception as transcribe_error:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"Transcription error: {str(transcribe_error)}"}
                    )
            
            elif text:
                # Text from form data
                user_text = text.strip()
                if not user_text:
                    return JSONResponse(status_code=400, content={"error": "Text is empty"})
                print(f" Form text query: '{user_text[:100]}...'")
            
            else:
                return JSONResponse(status_code=400, content={"error": "No audio or text input provided"})

        # Send to Gemini LLM (common for both Day 8 & 9)
        print(f" Sending to Gemini...")
        
        gemini_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
        gemini_headers = {"Content-Type": "application/json"}
        gemini_params = {"key": GEMINI_API_KEY}
        gemini_payload = {
            "contents": [
                {
                    "parts": [{"text": user_text}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }

        try:
            gemini_response = requests.post(
                gemini_url, 
                headers=gemini_headers, 
                params=gemini_params, 
                json=gemini_payload,
                timeout=30
            )
            gemini_response.raise_for_status()
            gemini_result = gemini_response.json()

            ai_text = gemini_result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            if not ai_text:
                return JSONResponse(status_code=500, content={"error": "Empty response from Gemini LLM"})
            
            print(f" Gemini response: '{ai_text[:100]}...'")

        except requests.exceptions.RequestException as gemini_error:
            return JSONResponse(status_code=500, content={"error": f"Gemini API error: {str(gemini_error)}"})

        # For Day 8 (text input): Return only text response
        if not is_voice_input:
            return {"response": ai_text}

        # For Day 9 (voice input): Also generate voice response
        if not MURF_API_KEY:
            return {
                "transcription": transcription,
                "response": ai_text,
                "audio_url": None,
                "error": "MURF_API_KEY not set - text response only"
            }

        print(" Generating voice response...")
        
        murf_headers = {
            "Authorization": f"Bearer {MURF_API_KEY}",
            "Content-Type": "application/json",
            "api-key": f"{MURF_API_KEY}"
        }
        murf_body = {
            "voiceId": "en-UK-hazel",
            "text": ai_text
        }

        try:
            murf_response = requests.post(
                "https://api.murf.ai/v1/speech/generate", 
                json=murf_body, 
                headers=murf_headers,
                timeout=30
            )
            murf_response.raise_for_status()
            
            audio_url = murf_response.json().get("audioFile")
            
            if not audio_url:
                return {
                    "transcription": transcription,
                    "response": ai_text,
                    "audio_url": None,
                    "error": "Voice generation failed"
                }
            
            print(f" Voice generated: {audio_url}")
            
            return {
                "transcription": transcription,
                "response": ai_text,
                "audio_url": audio_url
            }

        except requests.exceptions.RequestException as murf_error:
            return {
                "transcription": transcription,
                "response": ai_text,
                "audio_url": None,
                "error": f"Voice generation failed: {str(murf_error)}"
            }

    except Exception as e:
        print(f" Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Pipeline error: {str(e)}"})


# ---------- DAY 10: Chat History with Audio Input ----------
@app.post("/agent/chat/{session_id}")
async def agent_chat(session_id: str, file: UploadFile = File(...)):
    try:
        #  Save uploaded audio temporarily
        temp_path = f"temp_{session_id}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        #  Transcribe with AssemblyAI
        transcript = transcriber.transcribe(temp_path)
        os.remove(temp_path)

        if not transcript.text or transcript.status != "completed":
            return JSONResponse(status_code=500, content={"error": "Transcription failed"})

        user_text = transcript.text.strip()
        print(f"User transcription for session {session_id}: {user_text}")

        #  Initialize chat history if new session
        if session_id not in chat_history:
            chat_history[session_id] = []

        #  Append user message to history
        chat_history[session_id].append({"role": "user", "content": user_text})

        #  Prepare full conversation for Gemini
        conversation_text = ""
        for msg in chat_history[session_id]:
            prefix = "User:" if msg["role"] == "user" else "Assistant:"
            conversation_text += f"{prefix} {msg['content']}\n"
        
        print(f"Full prompt for LLM:\n{conversation_text}")

        #  Send to Gemini API
        if not GEMINI_API_KEY:
            return JSONResponse(status_code=500, content={"error": "GEMINI_API_KEY not set"})
        
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        payload = {
            "contents": [
                {
                    "parts": [{"text": conversation_text}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }

        r = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()

        ai_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not ai_text:
            return JSONResponse(status_code=500, content={"error": "Empty response from Gemini"})

        print(f"AI response: {ai_text}")

        #  Append assistant's reply to history
        chat_history[session_id].append({"role": "assistant", "content": ai_text})

        #  Send AI text to Murf for TTS
        if not MURF_API_KEY:
            return JSONResponse(status_code=500, content={"error": "MURF_API_KEY not set"})
            
        murf_headers = {
            "Authorization": f"Bearer {MURF_API_KEY}",
            "Content-Type": "application/json",
            "api-key": f"{MURF_API_KEY}"
        }
        murf_body = {
            "voiceId": "en-UK-hazel",  # Change if needed
            "text": ai_text
        }
        murf_response = requests.post("https://api.murf.ai/v1/speech/generate", json=murf_body, headers=murf_headers, timeout=30)

        if murf_response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": murf_response.text})

        audio_url = murf_response.json().get("audioFile")

        # Return the response with the audio URL
        return {
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_url": audio_url,
            "chat_history": chat_history[session_id]
        }

    except Exception as e:
        print(f"Error in /agent/chat/{session_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Health Check ----------
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API keys and services"""
    status = {
        "status": "healthy",
        "services": {
            "murf": "✅" if MURF_API_KEY else "❌ Missing MURF_API_KEY",
            "assemblyai": "✅" if ASSEMBLYAI_API_KEY else "❌ Missing ASSEMBLYAI_API_KEY", 
            "gemini": "✅" if GEMINI_API_KEY else "❌ Missing GEMINI_API_KEY"
        }
    }
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)