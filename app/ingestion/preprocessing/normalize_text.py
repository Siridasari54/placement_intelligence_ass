import re

def normalize_whitespace(text: str) -> str:
    """Collapses consecutive spaces and newlines into single ones."""
    if not text:
        return ""
    # Collapse multiple newlines down to max two, and multiple spaces to one
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()
