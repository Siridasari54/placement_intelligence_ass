"""Voice input transcription module using Groq Whisper API."""

import logging
from typing import Optional, Tuple
from groq import Groq
from config.settings import settings

logger = logging.getLogger(__name__)

def transcribe_audio(audio_file, language: str) -> Tuple[str, str]:
    """Transcribes audio using Groq Whisper API and normalizes multilingual queries.
    
    Args:
        audio_file: Streamlit audio input file-like object
        language: Selected language ("English", "Telugu", or "Hindi")
        
    Returns:
        Tuple of (normalized_query, raw_transcript) - both strings, or ("", "") on error
    """
    if not audio_file:
        return "", ""
        
    # Map language names to ISO 639-1 language codes
    language_map = {
        "English": "en",
        "Telugu": "te",
        "Hindi": "hi"
    }
    lang_code = language_map.get(language, "en")
    
    try:
        # Initialize Groq client with the API key from settings
        client = Groq(api_key=settings.groq_api_key)
        
        # Read file bytes from Streamlit UploadedFile/BytesIO object
        audio_bytes = audio_file.read()
        
        # Whisper accepts common extensions like .wav, .webm, etc.
        filename = "audio.wav"
        if hasattr(audio_file, "name") and audio_file.name:
            filename = audio_file.name
            
        logger.info(f"Sending audio file {filename} ({len(audio_bytes)} bytes) to Groq Whisper API with language code: {lang_code}")
        
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            language=lang_code
        )
        
        transcript_text = transcription.text.strip()
        logger.info(f"Transcription successful. Transcript length: {len(transcript_text)} characters.")
        logger.info(f"Raw Transcript: {transcript_text}")
        
        # Apply query normalization for multilingual queries
        from core.query_normalization import normalize_multilingual_query, is_multilingual_query
        
        normalized_query = transcript_text
        if is_multilingual_query(transcript_text) or language in ["Telugu", "Hindi"]:
            logger.info("Detected multilingual query, applying normalization...")
            normalized_query = normalize_multilingual_query(transcript_text)
            logger.info(f"Normalized Query: {normalized_query}")
        else:
            logger.info("Query appears to be English, skipping normalization.")
        
        return normalized_query, transcript_text
        
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}", exc_info=True)
        return "", ""

def transcribe_audio_via_api(audio_file, language: str) -> Tuple[str, str]:
    """Transcribes audio file using local FastAPI transcription endpoint /transcribe.
    
    Falls back to local direct transcription if the API endpoint is unavailable.
    
    Returns:
        Tuple of (normalized_query, raw_transcript) - both strings, or ("", "") on error
    """
    import requests
    
    if not audio_file:
        return "", ""
        
    try:
        url = "http://localhost:8000/transcribe"
        filename = "audio.wav"
        if hasattr(audio_file, "name") and audio_file.name:
            filename = audio_file.name
            
        audio_bytes = audio_file.read() if hasattr(audio_file, "read") else audio_file.getvalue()
        
        # Reset pointer if it's a file-like object
        if hasattr(audio_file, "seek"):
            audio_file.seek(0)
            
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"language": language}
        
        logger.info(f"Uploading audio file {filename} to FastAPI endpoint {url} with language: {language}")
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            transcript = response.json().get("transcript", "").strip()
            logger.info(f"API transcription successful. Transcript length: {len(transcript)} characters.")
            logger.info(f"Raw Transcript: {transcript}")
            
            # Apply query normalization for multilingual queries
            from core.query_normalization import normalize_multilingual_query, is_multilingual_query
            
            normalized_query = transcript
            if is_multilingual_query(transcript) or language in ["Telugu", "Hindi"]:
                logger.info("Detected multilingual query, applying normalization...")
                normalized_query = normalize_multilingual_query(transcript)
                logger.info(f"Normalized Query: {normalized_query}")
            else:
                logger.info("Query appears to be English, skipping normalization.")
            
            return normalized_query, transcript
        else:
            logger.error(f"API transcription failed with status {response.status_code}: {response.text}")
            # Fallback to local Groq client transcription if FastAPI server is down/fails
            logger.info("Falling back to local direct transcription...")
            return transcribe_audio(audio_file, language)
            
    except Exception as e:
        logger.error(f"Error calling /transcribe API: {e}. Falling back to direct transcription...", exc_info=True)
        # Reset pointer if it's a file-like object
        if hasattr(audio_file, "seek"):
            audio_file.seek(0)
        return transcribe_audio(audio_file, language)
