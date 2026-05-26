from typing import List, Dict, Any
from app.utils.citation_builder import build_citations_list

class CitationService:
    @staticmethod
    def get_citations(chunks: List[Dict[str, Any]]) -> List[str]:
        return build_citations_list(chunks)
