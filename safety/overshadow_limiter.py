"""AIMD binary-search context cap to reduce overshadowing."""

from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class OvershadowLimiter:
    """AIMD binary-search context cap to reduce overshadowing of important chunks."""
    
    def __init__(self, max_context: int = 8192):
        """Initialize the overshadow limiter.
        
        Args:
            max_context: Maximum context window size in characters
        """
        self.max_context = max_context
        logger.info(f"OvershadowLimiter initialized with max_context={max_context}")
    
    def limit_context(self, documents: List[Document]) -> List[Document]:
        """Limit context using binary search to find optimal size.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Limited list of Document objects
        """
        logger.info(f"Limiting context for {len(documents)} documents")
        
        if not documents:
            return []
        
        # Binary search for optimal context size
        left, right = 0, len(documents)
        best_k = 0
        
        while left <= right:
            mid = (left + right) // 2
            total_chars = sum(len(doc.page_content) for doc in documents[:mid])
            
            if total_chars <= self.max_context:
                best_k = mid
                left = mid + 1
            else:
                right = mid - 1
        
        limited = documents[:best_k]
        logger.info(f"Limited context to {len(limited)} documents")
        return limited
