import re
from typing import Dict, Any
from app.utils.constants import COMPANIES

class MetadataParser:
    @staticmethod
    def extract_metadata(text: str, filename: str = "") -> Dict[str, Any]:
        """Deduce metadata fields from text content and file source."""
        metadata = {
            "company": "General",
            "section": "general",
            "source": "official",
            "conflict": False
        }
        
        text_lower = text.lower()
        
        # Match company
        for company in COMPANIES:
            # Check for exact boundary match to avoid substring false positives (e.g. 'Intel' in 'Intelligence')
            pattern = rf"\b{re.escape(company.lower())}\b"
            if re.search(pattern, text_lower):
                metadata["company"] = company
                break
                
        # Deduce section
        if "eligibility" in text_lower or "cutoff" in text_lower:
            metadata["section"] = "eligibility"
        elif "interview" in text_lower or "round details" in text_lower or "selection process" in text_lower:
            metadata["section"] = "interview"
        elif "hiring distribution" in text_lower or "role distribution" in text_lower:
            metadata["section"] = "hiring"
        elif "trend" in text_lower or "2021" in text_lower or "2022" in text_lower or "2023" in text_lower or "2024" in text_lower:
            metadata["section"] = "trend"
            # Extract year if possible
            years = re.findall(r"\b(202[1-4])\b", text)
            if years:
                metadata["year"] = int(years[0])
        elif "conflict" in text_lower or "portal" in text_lower or "unofficial" in text_lower:
            metadata["section"] = "conflict"
            metadata["conflict"] = True
            metadata["source"] = "portal"
            
        if filename:
            metadata["source_file"] = os.path.basename(filename)
            
        return metadata
