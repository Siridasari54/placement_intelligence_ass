"""State definition for the Placement Graph."""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.documents import Document
import operator


class PlacementState(TypedDict):
    """State for the placement intelligence graph."""
    
    query: str
    original_query: str
    documents: List[Document]
    route: str
    confidence: float
    fallback: bool
    answer: str
    retry_count: int
    is_comparison: bool
    comparison_entities: List[str]
