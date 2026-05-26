from typing import List, Dict, Any

def build_citation(metadata: Dict[str, Any], index: int) -> str:
    """Builds a formatted markdown citation link for a chunk."""
    company = metadata.get("company", "General")
    section = metadata.get("section", "unknown")
    source = metadata.get("source", "official")
    year = metadata.get("year", "")
    
    label = f"{company} {section.capitalize()}"
    if year:
        label += f" ({year})"
    if source != "official":
        label += f" [{source.upper()}]"
        
    # We can link to the source file or just output a structured citation reference
    # Let's link to the processed file or database record
    file_path = f"file:///c:/Users/sirid/OneDrive/Desktop/placement_rag_sys/data/processed/{section}_data.json"
    
    return f"[[{index}] {label}]({file_path})"

def build_citations_list(chunks: List[Dict[str, Any]]) -> List[str]:
    """Compiles list of citations for a set of retrieved chunks."""
    citations = []
    seen = set()
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        # Create a unique key to avoid duplicate citations
        key = (meta.get("company"), meta.get("section"), meta.get("year"), meta.get("source"))
        if key not in seen:
            seen.add(key)
            citations.append(build_citation(meta, len(citations) + 1))
    return citations
