import os
import sys
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# Add parent directory to path to import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Placement Assistant Audio Transcription API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "running", "service": "voice_transcription"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("English")):
    """Transcribes an uploaded audio file using Groq Whisper API."""
    logger.info(f"Received transcription request for file: {file.filename}, language: {language}")
    
    # Map language names to ISO 639-1 language codes
    language_map = {
        "English": "en",
        "Telugu": "te",
        "Hindi": "hi"
    }
    lang_code = language_map.get(language, "en")
    
    try:
        # Get API key from settings
        api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable is required")
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server")
            
        client = Groq(api_key=api_key)
        
        # Read the uploaded file bytes
        audio_bytes = await file.read()
        filename = file.filename or "audio.wav"
        
        logger.info(f"Uploading file '{filename}' ({len(audio_bytes)} bytes) to Groq Whisper model: whisper-large-v3-turbo")
        
        # Call Groq Whisper API
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            language=lang_code
        )
        
        transcript_text = transcription.text.strip()
        logger.info(f"Transcription successful. Text: {transcript_text}")
        return {"transcript": transcript_text}
        
    except Exception as e:
        logger.error(f"Error during API audio transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
