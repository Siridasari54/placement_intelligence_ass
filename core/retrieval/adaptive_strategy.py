"""Adaptive Retrieval Strategy for dynamic mode selection based on query characteristics."""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Retrieval mode types."""
    SEMANTIC_HEAVY = "semantic_heavy"  # 80% vector, 20% BM25
    KEYWORD_HEAVY = "keyword_heavy"  # 20% vector, 80% BM25
    BALANCED_HYBRID = "balanced_hybrid"  # 50% vector, 50% BM25
    METADATA_FIRST = "metadata_first"  # Filter by metadata first
    MULTI_HOP = "multi_hop"  # Iterative retrieval
    DENSE_ONLY = "dense_only"  # Pure vector search
    SPARSE_ONLY = "sparse_only"  # Pure BM25 search


@dataclass
class ModeWeights:
    """Weights for hybrid retrieval modes."""
    vector_weight: float
    bm25_weight: float
    metadata_weight: float


class AdaptiveRetrievalStrategy:
    """Adaptive retrieval strategy that dynamically selects mode based on query."""
    
    def __init__(self):
        """Initialize the adaptive retrieval strategy."""
        self.mode_weights = {
            RetrievalMode.SEMANTIC_HEAVY: ModeWeights(0.8, 0.2, 0.0),
            RetrievalMode.KEYWORD_HEAVY: ModeWeights(0.2, 0.8, 0.0),
            RetrievalMode.BALANCED_HYBRID: ModeWeights(0.5, 0.5, 0.0),
            RetrievalMode.METADATA_FIRST: ModeWeights(0.3, 0.3, 0.4),
            RetrievalMode.MULTI_HOP: ModeWeights(0.6, 0.4, 0.0),
            RetrievalMode.DENSE_ONLY: ModeWeights(1.0, 0.0, 0.0),
            RetrievalMode.SPARSE_ONLY: ModeWeights(0.0, 1.0, 0.0)
        }
        
        # Query characteristics for mode selection
        self.semantic_indicators = [
            r'\b(meaning|concept|understand|explain|describe)\b',
            r'\b(similar|related|associated)\b',
            r'\b(what is|how does|why)\b'
        ]
        
        self.keyword_indicators = [
            r'\b(specific|exact|precise)\b',
            r'\b(number|count|amount)\b',
            r'\b(company|name|title)\b'
        ]
        
        self.metadata_indicators = [
            r'\b(company|organization)\b',
            r'\b(year|date|time)\b',
            r'\b(package|salary|lpa)\b'
        ]
        
        self.multi_hop_indicators = [
            r'\b(and|then|also|additionally)\b',
            r'\b(compare|difference|versus)\b',
            r'\b(trend|change|over time)\b'
        ]
        
        logger.info("AdaptiveRetrievalStrategy initialized")
    
    def select_mode(
        self,
        query: str,
        metadata_filters: Dict[str, Any] = None
    ) -> RetrievalMode:
        """Select appropriate retrieval mode based on query characteristics.
        
        Args:
            query: User query
            metadata_filters: Optional metadata filters
            
        Returns:
            Selected retrieval mode
        """
        logger.info(f"Selecting retrieval mode for query: {query}")
        
        # Calculate mode scores
        scores = {
            RetrievalMode.SEMANTIC_HEAVY: self._calculate_semantic_score(query),
            RetrievalMode.KEYWORD_HEAVY: self._calculate_keyword_score(query),
            RetrievalMode.METADATA_FIRST: self._calculate_metadata_score(query, metadata_filters),
            RetrievalMode.MULTI_HOP: self._calculate_multi_hop_score(query)
        }
        
        # Select mode with highest score
        selected_mode = max(scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Selected mode: {selected_mode.value} (score: {scores[selected_mode]:.2f})")
        return selected_mode
    
    def get_weights(self, mode: RetrievalMode) -> ModeWeights:
        """Get weights for a specific retrieval mode.
        
        Args:
            mode: Retrieval mode
            
        Returns:
            Mode weights
        """
        return self.mode_weights.get(mode, self.mode_weights[RetrievalMode.BALANCED_HYBRID])
    
    def _calculate_semantic_score(self, query: str) -> float:
        """Calculate semantic-heavy score.
        
        Args:
            query: Query text
            
        Returns:
            Semantic score between 0 and 1
        """
        score = 0.0
        query_lower = query.lower()
        
        for pattern in self.semantic_indicators:
            if re.search(pattern, query_lower):
                score += 0.3
        
        # Boost for longer, more descriptive queries
        if len(query.split()) > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_keyword_score(self, query: str) -> float:
        """Calculate keyword-heavy score.
        
        Args:
            query: Query text
            
        Returns:
            Keyword score between 0 and 1
        """
        score = 0.0
        query_lower = query.lower()
        
        for pattern in self.keyword_indicators:
            if re.search(pattern, query_lower):
                score += 0.3
        
        # Boost for short, specific queries
        if len(query.split()) < 5:
            score += 0.2
        
        # Boost for queries with specific terms (capitalized words)
        specific_terms = [word for word in query.split() if word[0].isupper()]
        if len(specific_terms) > 0:
            score += 0.1 * min(len(specific_terms), 3)
        
        return min(score, 1.0)
    
    def _calculate_metadata_score(
        self,
        query: str,
        metadata_filters: Dict[str, Any] = None
    ) -> float:
        """Calculate metadata-first score.
        
        Args:
            query: Query text
            metadata_filters: Optional metadata filters
            
        Returns:
            Metadata score between 0 and 1
        """
        score = 0.0
        query_lower = query.lower()
        
        for pattern in self.metadata_indicators:
            if re.search(pattern, query_lower):
                score += 0.3
        
        # Boost if metadata filters are provided
        if metadata_filters:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_multi_hop_score(self, query: str) -> float:
        """Calculate multi-hop score.
        
        Args:
            query: Query text
            
        Returns:
            Multi-hop score between 0 and 1
        """
        score = 0.0
        query_lower = query.lower()
        
        for pattern in self.multi_hop_indicators:
            if re.search(pattern, query_lower):
                score += 0.3
        
        # Boost for complex queries with multiple clauses
        if query.count(',') >= 2 or query.count('and') >= 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def adapt_weights(
        self,
        mode: RetrievalMode,
        performance_feedback: Dict[str, float]
    ) -> ModeWeights:
        """Adapt weights based on performance feedback.
        
        Args:
            mode: Current retrieval mode
            performance_feedback: Performance metrics
            
        Returns:
            Adapted mode weights
        """
        base_weights = self.get_weights(mode)
        
        # Adjust weights based on feedback
        if performance_feedback.get("vector_precision", 0) > 0.8:
            base_weights.vector_weight = min(base_weights.vector_weight + 0.1, 1.0)
            base_weights.bm25_weight = max(base_weights.bm25_weight - 0.1, 0.0)
        
        if performance_feedback.get("bm25_precision", 0) > 0.8:
            base_weights.bm25_weight = min(base_weights.bm25_weight + 0.1, 1.0)
            base_weights.vector_weight = max(base_weights.vector_weight - 0.1, 0.0)
        
        # Normalize weights
        total = base_weights.vector_weight + base_weights.bm25_weight + base_weights.metadata_weight
        if total > 0:
            base_weights.vector_weight /= total
            base_weights.bm25_weight /= total
            base_weights.metadata_weight /= total
        
        logger.info(f"Adapted weights: vector={base_weights.vector_weight:.2f}, bm25={base_weights.bm25_weight:.2f}")
        return base_weights


class DynamicRetriever:
    """Dynamic retriever that adapts strategy based on query and performance."""
    
    def __init__(self, adaptive_strategy: AdaptiveRetrievalStrategy):
        """Initialize the dynamic retriever.
        
        Args:
            adaptive_strategy: Adaptive retrieval strategy
        """
        self.strategy = adaptive_strategy
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info("DynamicRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        vector_retriever,
        bm25_retriever,
        k: int = 5,
        metadata_filters: Dict[str, Any] = None
    ) -> Tuple[List, Dict[str, Any]]:
        """Retrieve documents using adaptive strategy.
        
        Args:
            query: User query
            vector_retriever: Vector retriever instance
            bm25_retriever: BM25 retriever instance
            k: Number of documents to retrieve
            metadata_filters: Optional metadata filters
            
        Returns:
            Tuple of (retrieved documents, retrieval metadata)
        """
        logger.info(f"Dynamic retrieval for query: {query}")
        
        # Select retrieval mode
        mode = self.strategy.select_mode(query, metadata_filters)
        
        # Get weights for selected mode
        weights = self.strategy.get_weights(mode)
        
        # Perform retrieval based on mode
        if mode == RetrievalMode.DENSE_ONLY:
            documents = vector_retriever.retrieve(query, k=k)
        elif mode == RetrievalMode.SPARSE_ONLY:
            documents = bm25_retriever.retrieve(query, k=k)
        else:
            # Hybrid retrieval with weights
            vector_docs = vector_retriever.retrieve(query, k=k)
            bm25_docs = bm25_retriever.retrieve(query, k=k)
            documents = self._fuse_results(vector_docs, bm25_docs, weights)
        
        # Apply metadata filters if present
        if metadata_filters and mode == RetrievalMode.METADATA_FIRST:
            documents = self._apply_metadata_filters(documents, metadata_filters)
        
        retrieval_metadata = {
            "mode": mode.value,
            "weights": {
                "vector": weights.vector_weight,
                "bm25": weights.bm25_weight,
                "metadata": weights.metadata_weight
            },
            "documents_retrieved": len(documents)
        }
        
        logger.info(f"Retrieval complete with mode: {mode.value}")
        return documents, retrieval_metadata
    
    def _fuse_results(
        self,
        vector_docs: List,
        bm25_docs: List,
        weights: ModeWeights
    ) -> List:
        """Fuse results from vector and BM25 retrieval.
        
        Args:
            vector_docs: Vector retrieval results
            bm25_docs: BM25 retrieval results
            weights: Fusion weights
            
        Returns:
            Fused results
        """
        # Simple weighted fusion
        fused = []
        
        # Combine and score
        all_docs = []
        for i, doc in enumerate(vector_docs):
            all_docs.append((doc, weights.vector_weight * (1.0 / (i + 1))))
        
        for i, doc in enumerate(bm25_docs):
            all_docs.append((doc, weights.bm25_weight * (1.0 / (i + 1))))
        
        # Sort by score and deduplicate
        seen = set()
        for doc, score in sorted(all_docs, key=lambda x: x[1], reverse=True):
            doc_id = str(doc)
            if doc_id not in seen:
                fused.append(doc)
                seen.add(doc_id)
        
        return fused
    
    def _apply_metadata_filters(self, documents: List, filters: Dict[str, Any]) -> List:
        """Apply metadata filters to documents.
        
        Args:
            documents: List of documents
            filters: Metadata filters
            
        Returns:
            Filtered documents
        """
        filtered = []
        
        for doc in documents:
            match = True
            doc_metadata = getattr(doc, 'metadata', {})
            
            for key, value in filters.items():
                if key not in doc_metadata or doc_metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered.append(doc)
        
        logger.info(f"Filtered {len(documents)} -> {len(filtered)} documents")
        return filtered
    
    def record_performance(self, metadata: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Record performance for adaptive learning.
        
        Args:
            metadata: Retrieval metadata
            metrics: Performance metrics
        """
        performance_entry = {
            "mode": metadata.get("mode"),
            "weights": metadata.get("weights"),
            "metrics": metrics,
            "timestamp": str(datetime.now())
        }
        
        self.performance_history.append(performance_entry)
        
        # Adapt strategy based on performance
        if len(self.performance_history) > 10:
            self.strategy.adapt_weights(
                RetrievalMode(metadata.get("mode")),
                metrics
            )
        
        logger.info("Performance recorded and strategy adapted")
