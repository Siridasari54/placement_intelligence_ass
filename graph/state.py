"""State definition for the Placement Graph."""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.documents import Document
import operator


class PlacementState(TypedDict):
    """State for the placement intelligence graph."""
    
    query: str
    documents: List[Document]
    route: str
    confidence: float
    fallback: bool
    answer: str
