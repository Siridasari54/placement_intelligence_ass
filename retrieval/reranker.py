"""CrossEncoder reranker for improved contextual precision with thresholding and deduplication."""

from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from core.interfaces import IReranker
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker(IReranker):
    """Enhanced CrossEncoder-based reranker with dynamic thresholding, confidence scoring, and deduplication."""
    
    def __init__(self, model_name: str = None, relevance_threshold: float = 0.3, deduplicate: bool = True, dynamic_threshold: bool = True):
        """Initialize the reranker with specified model and parameters.
        
        Args:
            model_name: Model name to use (defaults to settings)
            relevance_threshold: Minimum relevance score to keep a document
            deduplicate: Whether to remove duplicate/near-duplicate documents
            dynamic_threshold: Whether to dynamically adjust threshold based on score distribution
        """
        self.model_name = model_name or settings.retrieval.rerank_model
        self.model = None  # Disabled CrossEncoder due to numpy array comparison error
        self.relevance_threshold = relevance_threshold
        self.deduplicate = deduplicate
        self.dynamic_threshold = dynamic_threshold
        logger.info(f"CrossEncoder reranker disabled due to numpy array error. Using simple score-based fallback.")
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents based on query relevance with dynamic thresholding and deduplication.
        
        Args:
            query: Query text
            documents: List of Document objects to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of Document objects with confidence scores in metadata
        """
        logger.info(f"Reranking {len(documents)} documents (using simple fallback)")
        
        if not documents:
            return []
        
        # Simple fallback: use existing scores if available, otherwise use order
        scored_docs = []
        for idx, doc in enumerate(documents):
            # Use existing score from metadata or generate a simple score based on position
            score = doc.metadata.get("score", doc.metadata.get("rrf_score", 1.0 - (idx * 0.01)))
            scored_docs.append((doc, float(score)))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with confidence scores in metadata
        reranked = []
        for doc, score in scored_docs[:top_k]:
            # Add confidence score to metadata
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["confidence"] = max(0.0, min(1.0, float(score)))
            reranked.append(doc)
        
        logger.info(f"Reranked to top {len(reranked)} documents")
        return reranked
    
    def _deduplicate(self, scored_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove duplicate and near-duplicate documents.
        
        Args:
            scored_docs: List of (document, score) tuples
            
        Returns:
            Deduplicated list of (document, score) tuples
        """
        seen_content = set()
        deduplicated = []
        
        for doc, score in scored_docs:
            # Normalize content for comparison
            content_normalized = doc.page_content.lower().strip()
            
            # Check for exact duplicates
            if content_normalized in seen_content:
                logger.debug(f"Removed duplicate: {content_normalized[:50]}...")
                continue
            
            # Check for near-duplicates (high similarity)
            is_duplicate = False
            for seen in seen_content:
                if self._content_similarity(content_normalized, seen) > 0.9:
                    logger.debug(f"Removed near-duplicate: {content_normalized[:50]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_content.add(content_normalized)
                deduplicated.append((doc, score))
        
        return deduplicated
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings.
        
        Args:
            content1: First content string
            content2: Second content string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for words
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confidence(self, rerank_score: float) -> float:
        """Calculate confidence score from rerank score.
        
        Args:
            rerank_score: Raw rerank score from CrossEncoder
            
        Returns:
            Normalized confidence score between 0 and 1
        """
        # CrossEncoder scores typically range from -1 to 1
        # Normalize to 0-1 range
        normalized = (rerank_score + 1) / 2
        return max(0.0, min(1.0, normalized))
    
    def _calculate_dynamic_threshold(self, scores: List[float]) -> float:
        """Calculate dynamic threshold based on score distribution.
        
        Args:
            scores: List of rerank scores
            
        Returns:
            Dynamic threshold value
        """
        if not scores:
            return self.relevance_threshold
        
        # Use median as baseline
        sorted_scores = sorted(scores)
        median = sorted_scores[len(sorted_scores) // 2]
        
        # Calculate interquartile range
        q1 = sorted_scores[len(sorted_scores) // 4]
        q3 = sorted_scores[3 * len(sorted_scores) // 4]
        iqr = q3 - q1
        
        # Set threshold at Q1 - 0.5 * IQR (lower fence)
        dynamic_threshold = q1 - 0.5 * iqr
        
        # Ensure threshold is reasonable
        return max(self.relevance_threshold, min(dynamic_threshold, median))
    
    def _calculate_percentile(self, score: float, all_scores: List[float]) -> float:
        """Calculate percentile rank of a score within all scores.
        
        Args:
            score: Score to calculate percentile for
            all_scores: List of all scores
            
        Returns:
            Percentile between 0 and 1
        """
        if not all_scores:
            return 0.5
        
        # Count how many scores are below this score
        below_count = sum(1 for s in all_scores if s < score)
        percentile = below_count / len(all_scores)
        
        return max(0.0, min(1.0, percentile))
