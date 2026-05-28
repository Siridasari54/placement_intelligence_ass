"""Context refiner for noise filtering and deduplication."""

from typing import List, Tuple
from langchain_core.documents import Document
from core.interfaces import IRefiner
import logging

logger = logging.getLogger(__name__)


class ContextRefiner(IRefiner):
    """Context refiner for filtering noise and reducing repetitive chunks."""
    
    def __init__(self, min_length: int = 50, max_duplicates: int = 2):
        """Initialize the context refiner.
        
        Args:
            min_length: Minimum chunk length to keep
            max_duplicates: Maximum allowed duplicate chunks
        """
        self.min_length = min_length
        self.max_duplicates = max_duplicates
        logger.info("ContextRefiner initialized")
    
    def refine(self, documents: List[Document]) -> List[Document]:
        """Refine and filter documents for better context.
        
        Args:
            documents: List of Document objects to refine
            
        Returns:
            Refined list of Document objects
        """
        logger.info(f"Refining {len(documents)} documents")
        
        # Filter by length
        filtered = [
            doc for doc in documents
            if len(doc.page_content.strip()) >= self.min_length
        ]
        
        # Remove duplicates
        seen = set()
        deduplicated = []
        for doc in filtered:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars for hashing
            if content_hash not in seen:
                deduplicated.append(doc)
                seen.add(content_hash)
        
        logger.info(f"Refined to {len(deduplicated)} documents")
        return deduplicated


class OvershadowLimiter:
    """AIMD binary-search context cap to reduce overshadowing."""
    
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
        logger.info(f"Limited context to {len(limited)} documents ({best_k} chars)")
        return limited
