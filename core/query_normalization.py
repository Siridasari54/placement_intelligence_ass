"""Query Normalization Layer for multilingual voice input processing.

This module provides functionality to normalize multilingual (Telugu-English, Hindi-English)
voice transcripts into natural English queries before tool routing.
"""

import logging
from typing import Optional
from groq import Groq
from config.settings import settings

logger = logging.getLogger(__name__)


def normalize_multilingual_query(transcript: str) -> str:
    """Normalize multilingual voice transcript into natural English query.
    
    This function uses the Groq LLM to convert Telugu-English and Hindi-English
    mixed queries into natural English, correct obvious transcription mistakes,
    preserve the original intent, and return only the normalized English query.
    
    Args:
        transcript: Raw Whisper transcript (may contain Telugu/Hindi words)
        
    Returns:
        Normalized English query string, or original transcript if normalization fails
        
    Examples:
        "Wipro CEO evaru" → "Who is the CEO of Wipro?"
        "Infosys package entha" → "What package does Infosys offer?"
        "TCS CGPA entha" → "What is the CGPA requirement for TCS?"
        "Google lo internship unda" → "Does Google offer internships?"
    """
    if not transcript or not transcript.strip():
        return transcript
    
    try:
        # Initialize Groq client
        client = Groq(api_key=settings.groq_api_key)
        
        # Build normalization prompt
        prompt = f"""You are a precise query normalizer for a college placement assistant.
Your task is to convert multilingual voice transcripts (Telugu-English, Hindi-English mixed)
into natural English queries while preserving the original intent.

Common Telugu/Hindi words in placement context:
- "evaru" = who
- "entha" = what/how much
- "ela" = how
- "enda" = which
- "unda" = is there/does it have
- "kavali" = need/required
- "elanti" = what kind of
- "ekkuva" = more/higher
- "takkuvaga" = less/lower
- "cheyyali" = should do/must
- "cheyyaledu" = didn't do
- "cheyyaledante" = if not done

Examples:
- "Wipro CEO evaru" → "Who is the CEO of Wipro?"
- "Infosys package entha" → "What package does Infosys offer?"
- "TCS CGPA entha" → "What is the CGPA requirement for TCS?"
- "Google lo internship unda" → "Does Google offer internships?"
- "Amazon salary elanti" → "What kind of salary does Amazon offer?"
- "Microsoft lo kavali" → "What are the requirements for Microsoft?"
- "Highest package ekkuva company" → "Which company offers the highest package?"
- "Backlog takkuvaga companies" → "Which companies allow fewer backlogs?"

Rules:
1. Convert the transcript to natural English
2. Correct obvious transcription mistakes (e.g., "Vipro" → "Wipro")
3. Preserve the original intent and question type
4. Return ONLY the normalized English query, no explanations
5. If the transcript is already in English, return it as-is (with minor corrections if needed)

Raw Transcript: "{transcript}"

Normalized Query:"""

        logger.info(f"Normalizing multilingual query: {transcript[:100]}...")
        
        response = client.chat.completions.create(
            model=settings.generation.model,
            messages=[
                {"role": "system", "content": "You are a precise query normalizer that outputs only the normalized English query."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=256
        )
        
        normalized_query = response.choices[0].message.content.strip()
        
        # Clean up any markdown formatting
        normalized_query = normalized_query.replace('```', '').replace('"', '').strip()
        
        logger.info(f"Normalized query: {normalized_query}")
        return normalized_query
        
    except Exception as e:
        logger.error(f"Error normalizing multilingual query: {e}", exc_info=True)
        # Return original transcript if normalization fails
        return transcript


def is_multilingual_query(query: str) -> bool:
    """Check if a query appears to be multilingual (contains Telugu/Hindi words).
    
    Args:
        query: Query string to check
        
    Returns:
        True if query appears to be multilingual, False otherwise
    """
    # Common Telugu/Hindi indicators
    multilingual_indicators = [
        'evaru', 'entha', 'ela', 'enda', 'unda', 'kavali', 'elanti',
        'ekkuva', 'takkuvaga', 'cheyyali', 'cheyyaledu', 'cheyyaledante',
        'lo', 'ki', 'ku', 'ni', 'di', 'du', 'lu', 'ru'
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in multilingual_indicators)
