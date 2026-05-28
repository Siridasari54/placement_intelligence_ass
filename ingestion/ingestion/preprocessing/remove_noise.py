import re

def remove_page_numbers_and_headers(text: str) -> str:
    """Removes common headers, footers, and page numbers from document text."""
    if not text:
        return ""
        
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # Detect page indicators like "Page 1 of 10" or "Section 1: ..." repeated
        if re.match(r"^page\s*\d+\s*(of\s*\d+)?$", stripped, re.IGNORECASE):
            continue
            
        if re.match(r"^svecw\s*·\s*department\s*of\s*information\s*technology.*", stripped, re.IGNORECASE):
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)
