import re
from typing import Dict, Any, List
from app.utils.constants import COMPANIES

class MetadataFiltering:
    @staticmethod
    def build_filter_from_query(query: str) -> Dict[str, Any]:
        """Inspects query text and builds a metadata pre-filtering dictionary."""
        query_lower = query.lower()
        filters = {}
        
        # 1. Match company
        matched_company = None
        for company in COMPANIES:
            # Check boundary match
            pattern = rf"\b{re.escape(company.lower())}\b"
            if re.search(pattern, query_lower):
                matched_company = company
                break
                
        if matched_company:
            filters["company"] = matched_company
            
        # 2. Match section
        if "interview" in query_lower or "round" in query_lower:
            filters["section"] = "interview"
        elif "eligibility" in query_lower or "cutoff" in query_lower or "backlog" in query_lower:
            filters["section"] = "eligibility"
        elif "trend" in query_lower or "timeline" in query_lower:
            filters["section"] = "trend"
        elif "hiring" in query_lower or "role" in query_lower or "intern" in query_lower or "analyst" in query_lower:
            filters["section"] = "hiring"
            
        return filters
