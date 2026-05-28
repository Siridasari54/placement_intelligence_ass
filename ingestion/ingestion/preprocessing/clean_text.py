import re

def clean_unicode_characters(text: str) -> str:
    """Cleans up smart quotes, unicode spaces, and common PDF garbles."""
    if not text:
        return ""
    # Standardize spaces and punctuation
    text = text.replace("\u201d", '"').replace("\u201c", '"')
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\xa0", " ")
    
    # Fix common garbled patterns
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text
