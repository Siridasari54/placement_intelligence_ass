"""Fallback guard for out-of-corpus detection and confidence scoring."""

from typing import List, Tuple
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class FallbackGuard:
    """Fallback guard for out-of-corpus detection and confidence scoring."""
    
    def __init__(self, threshold: float = 0.3):
        """Initialize the fallback guard.
        
        Args:
            threshold: Confidence threshold for fallback
        """
        self.threshold = threshold
        logger.info(f"FallbackGuard initialized with threshold {threshold}")
    
    def check_confidence(self, query: str, documents: List[Document]) -> float:
        """Calculate confidence score for retrieval results.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        logger.info("Calculating confidence score")
        
        if not documents:
            return 0.0
        
        # Simple confidence based on:
        # 1. Number of relevant documents
        # 2. Keyword overlap
        # 3. Source reliability
        
        query_keywords = set(query.lower().split())
        relevant_docs = 0
        source_bonus = 0
        
        for doc in documents:
            doc_text = doc.page_content.lower()
            
            # Keyword overlap
            if any(keyword in doc_text for keyword in query_keywords):
                relevant_docs += 1
            
            # Source reliability bonus
            if doc.metadata.get("source") == "official":
                source_bonus += 0.1
        
        # Calculate confidence
        relevance_ratio = relevant_docs / len(documents)
        confidence = min(relevance_ratio + source_bonus, 1.0)
        
        logger.info(f"Confidence score: {confidence:.2f}")
        return confidence
    
    def should_fallback(self, confidence: float) -> bool:
        """Determine if system should fallback to generic response.
        
        Args:
            confidence: Confidence score
            
        Returns:
            True if should fallback, False otherwise
        """
        should_fallback = confidence < self.threshold
        logger.info(f"Should fallback: {should_fallback}")
        return should_fallback
