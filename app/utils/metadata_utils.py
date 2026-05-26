from typing import Dict, Any

def create_metadata(
    company: str,
    section: str,
    year: int = None,
    source: str = "official",
    conflict: bool = False,
    additional: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generates standard metadata dictionary for chunks."""
    meta = {
        "company": company,
        "section": section,
        "source": source,
        "conflict": conflict
    }
    if year is not None:
        meta["year"] = year
    if additional:
        meta.update(additional)
    return meta

def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """Inspects text content to deduce metadata fields programmatically."""
    # Simple rule-based extraction
    text_lower = text.lower()
    
    # Identify company
    company = "Unknown"
    from app.utils.constants import COMPANIES
    for c in COMPANIES:
        if c.lower() in text_lower:
            company = c
            break
            
    # Identify section
    section = "general"
    if "eligibility" in text_lower or "cgpa" in text_lower or "backlog" in text_lower:
        section = "eligibility"
    elif "interview" in text_lower or "round" in text_lower or "hackerrank" in text_lower:
        section = "interview"
    elif "hiring" in text_lower or "analyst" in text_lower or "sde" in text_lower:
        section = "hiring"
    elif "trend" in text_lower or "2021" in text_lower or "2022" in text_lower or "2023" in text_lower or "2024" in text_lower:
        section = "trend"
        
    return create_metadata(company, section)
