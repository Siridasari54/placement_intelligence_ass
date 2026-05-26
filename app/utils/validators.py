import re

def validate_cgpa(cgpa: float) -> bool:
    """Validates if CGPA is in standard range (0 to 10.0)."""
    return 0.0 <= cgpa <= 10.0

def validate_backlogs(backlogs: int) -> bool:
    """Validates backlog count."""
    return backlogs >= 0

def clean_query(query: str) -> str:
    """Cleans search query by removing unwanted punctuation and trimming."""
    if not query:
        return ""
    # Retain alphanumeric, spaces, and dot/hyphen for packages
    cleaned = re.sub(r"[^\w\s\.\-]", "", query)
    return cleaned.strip()
