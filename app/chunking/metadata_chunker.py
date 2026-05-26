from typing import Dict, Any

class MetadataChunker:
    @staticmethod
    def enrich_chunk(chunk_text: str, metadata: Dict[str, Any]) -> str:
        """Prepends metadata tags to chunk content to improve embedding retrieval."""
        company = metadata.get("company", "")
        section = metadata.get("section", "")
        source = metadata.get("source", "official")
        year = metadata.get("year", "")
        
        prefix = f"[{company.upper()} | {section.upper()} | {source.upper()}"
        if year:
            prefix += f" | {year}"
        prefix += "] "
        
        return prefix + chunk_text
