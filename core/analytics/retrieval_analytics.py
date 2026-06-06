"""Retrieval Analytics System for monitoring and debugging retrieval performance."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
import os
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""
    query: str
    timestamp: datetime
    latency_ms: float
    documents_retrieved: int
    documents_reranked: int
    confidence_score: float
    reranking_scores: List[float]
    source_distribution: Dict[str, int]
    chunk_utilization: float
    token_usage: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalAnalytics:
    """Analytics system for tracking retrieval performance."""
    
    def __init__(self, analytics_file: str = "data/analytics.json"):
        """Initialize retrieval analytics.
        
        Args:
            analytics_file: Path to analytics data file
        """
        self.analytics_file = analytics_file
        self.metrics_history: List[RetrievalMetrics] = []
        self._load_analytics()
        
        logger.info("RetrievalAnalytics initialized")
    
    def _load_analytics(self) -> None:
        """Load analytics from file."""
        if os.path.exists(self.analytics_file):
            try:
                with open(self.analytics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = []
                    for metric in data:
                        if isinstance(metric.get("timestamp"), str):
                            try:
                                metric["timestamp"] = datetime.fromisoformat(metric["timestamp"])
                            except Exception:
                                metric["timestamp"] = datetime.now()
                        self.metrics_history.append(RetrievalMetrics(**metric))
            except Exception as e:
                logger.error(f"Error loading analytics file: {e}")
                self.metrics_history = []
        else:
            self.metrics_history = []
    
    def _save_analytics(self) -> None:
        """Save analytics to file."""
        os.makedirs(os.path.dirname(self.analytics_file), exist_ok=True)
        
        data = [
            {
                "query": m.query,
                "timestamp": m.timestamp.isoformat(),
                "latency_ms": m.latency_ms,
                "documents_retrieved": m.documents_retrieved,
                "documents_reranked": m.documents_reranked,
                "confidence_score": m.confidence_score,
                "reranking_scores": m.reranking_scores,
                "source_distribution": m.source_distribution,
                "chunk_utilization": m.chunk_utilization,
                "token_usage": m.token_usage,
                "metadata": m.metadata
            }
            for m in self.metrics_history
        ]
        
        with open(self.analytics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def track_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        reranked_docs: List[Document],
        confidence_score: float,
        reranking_scores: List[float],
        token_usage: int,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Track a retrieval operation.
        
        Args:
            query: Query text
            retrieved_docs: Retrieved documents
            reranked_docs: Reranked documents
            confidence_score: Confidence score
            reranking_scores: Reranking scores
            token_usage: Token usage
            metadata: Optional metadata
        """
        start_time = time.time()
        
        # Calculate source distribution
        source_distribution = self._calculate_source_distribution(reranked_docs)
        
        # Calculate chunk utilization
        chunk_utilization = self._calculate_chunk_utilization(reranked_docs)
        
        metric = RetrievalMetrics(
            query=query,
            timestamp=datetime.now(),
            latency_ms=0.0,  # Will be set after timing
            documents_retrieved=len(retrieved_docs),
            documents_reranked=len(reranked_docs),
            confidence_score=confidence_score,
            reranking_scores=reranking_scores,
            source_distribution=source_distribution,
            chunk_utilization=chunk_utilization,
            token_usage=token_usage,
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metric)
        self._save_analytics()
        
        logger.info(f"Tracked retrieval for query: {query[:50]}...")
    
    def _calculate_source_distribution(self, documents: List[Document]) -> Dict[str, int]:
        """Calculate distribution of document sources.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary of source counts
        """
        distribution = {}
        
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            distribution[source] = distribution.get(source, 0) + 1
        
        return distribution
    
    def _calculate_chunk_utilization(self, documents: List[Document]) -> float:
        """Calculate chunk utilization ratio.
        
        Args:
            documents: List of documents
            
        Returns:
            Utilization ratio between 0 and 1
        """
        if not documents:
            return 0.0
        
        # Calculate average chunk size utilization
        total_chars = sum(len(doc.page_content) for doc in documents)
        max_chars = len(documents) * 1000  # Assume 1000 chars max per chunk
        
        return min(total_chars / max_chars, 1.0)
    
    def get_aggregate_metrics(self, n: int = 100) -> Dict[str, Any]:
        """Get aggregate metrics for recent retrievals.
        
        Args:
            n: Number of recent retrievals to analyze
            
        Returns:
            Dictionary of aggregate metrics
        """
        recent_metrics = self.metrics_history[-n:]
        
        if not recent_metrics:
            return {}
        
        return {
            "total_retrievals": len(recent_metrics),
            "avg_latency_ms": sum(m.latency_ms for m in recent_metrics) / len(recent_metrics),
            "avg_documents_retrieved": sum(m.documents_retrieved for m in recent_metrics) / len(recent_metrics),
            "avg_confidence": sum(m.confidence_score for m in recent_metrics) / len(recent_metrics),
            "avg_chunk_utilization": sum(m.chunk_utilization for m in recent_metrics) / len(recent_metrics),
            "total_token_usage": sum(m.token_usage for m in recent_metrics),
            "source_distribution": self._aggregate_source_distribution(recent_metrics)
        }
    
    def _aggregate_source_distribution(self, metrics: List[RetrievalMetrics]) -> Dict[str, int]:
        """Aggregate source distribution across metrics.
        
        Args:
            metrics: List of metrics
            
        Returns:
            Aggregated source distribution
        """
        aggregated = {}
        
        for metric in metrics:
            for source, count in metric.source_distribution.items():
                aggregated[source] = aggregated.get(source, 0) + count
        
        return aggregated
    
    def get_latency_distribution(self, n: int = 100) -> Dict[str, float]:
        """Get latency distribution statistics.
        
        Args:
            n: Number of recent retrievals to analyze
            
        Returns:
            Dictionary of latency statistics
        """
        recent_metrics = self.metrics_history[-n:]
        
        if not recent_metrics:
            return {}
        
        latencies = [m.latency_ms for m in recent_metrics]
        
        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": sum(latencies) / len(latencies),
            "median": sorted(latencies)[len(latencies) // 2],
            "p95": sorted(latencies)[int(len(latencies) * 0.95)]
        }
    
    def get_confidence_trend(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get confidence score trend over time.
        
        Args:
            n: Number of recent retrievals to analyze
            
        Returns:
            List of confidence scores with timestamps
        """
        recent_metrics = self.metrics_history[-n:]
        
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "confidence": m.confidence_score
            }
            for m in recent_metrics
        ]
    
    def get_top_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent queries.
        
        Args:
            n: Number of top queries to return
            
        Returns:
            List of top queries with counts
        """
        query_counts = {}
        
        for metric in self.metrics_history:
            query = metric.query
            query_counts[query] = query_counts.get(query, 0) + 1
        
        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"query": query, "count": count}
            for query, count in sorted_queries[:n]
        ]
    
    def get_retrieval_debug_info(self, query: str) -> Optional[Dict[str, Any]]:
        """Get debug information for a specific query.
        
        Args:
            query: Query to look up
            
        Returns:
            Debug information if found, None otherwise
        """
        for metric in self.metrics_history:
            if metric.query == query:
                return {
                    "query": metric.query,
                    "timestamp": metric.timestamp.isoformat(),
                    "latency_ms": metric.latency_ms,
                    "documents_retrieved": metric.documents_retrieved,
                    "documents_reranked": metric.documents_reranked,
                    "confidence_score": metric.confidence_score,
                    "reranking_scores": metric.reranking_scores,
                    "source_distribution": metric.source_distribution,
                    "chunk_utilization": metric.chunk_utilization,
                    "token_usage": metric.token_usage,
                    "metadata": metric.metadata
                }
        
        return None
    
    def clear_analytics(self) -> None:
        """Clear all analytics data."""
        self.metrics_history = []
        self._save_analytics()
        logger.info("Analytics cleared")


class RetrievalDebugger:
    """Interactive debugger for retrieval operations."""
    
    def __init__(self, analytics: RetrievalAnalytics):
        """Initialize the retrieval debugger.
        
        Args:
            analytics: RetrievalAnalytics instance
        """
        self.analytics = analytics
        
        logger.info("RetrievalDebugger initialized")
    
    def debug_query(self, query: str) -> Dict[str, Any]:
        """Debug a specific query.
        
        Args:
            query: Query to debug
            
        Returns:
            Debug information
        """
        debug_info = self.analytics.get_retrieval_debug_info(query)
        
        if not debug_info:
            return {"error": "Query not found in analytics"}
        
        # Add analysis
        analysis = {
            "latency_analysis": self._analyze_latency(debug_info["latency_ms"]),
            "confidence_analysis": self._analyze_confidence(debug_info["confidence_score"]),
            "chunk_utilization_analysis": self._analyze_chunk_utilization(debug_info["chunk_utilization"]),
            "source_analysis": self._analyze_sources(debug_info["source_distribution"])
        }
        
        return {
            **debug_info,
            "analysis": analysis
        }
    
    def _analyze_latency(self, latency_ms: float) -> str:
        """Analyze latency.
        
        Args:
            latency_ms: Latency in milliseconds
            
        Returns:
            Analysis string
        """
        if latency_ms < 100:
            return "Excellent"
        elif latency_ms < 500:
            return "Good"
        elif latency_ms < 1000:
            return "Fair"
        else:
            return "Poor"
    
    def _analyze_confidence(self, confidence: float) -> str:
        """Analyze confidence score.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Analysis string
        """
        if confidence > 0.8:
            return "High confidence"
        elif confidence > 0.5:
            return "Medium confidence"
        else:
            return "Low confidence"
    
    def _analyze_chunk_utilization(self, utilization: float) -> str:
        """Analyze chunk utilization.
        
        Args:
            utilization: Utilization ratio
            
        Returns:
            Analysis string
        """
        if utilization > 0.8:
            return "High utilization"
        elif utilization > 0.5:
            return "Medium utilization"
        else:
            return "Low utilization"
    
    def _analyze_sources(self, distribution: Dict[str, int]) -> str:
        """Analyze source distribution.
        
        Args:
            distribution: Source distribution
            
        Returns:
            Analysis string
        """
        if not distribution:
            return "No sources"
        
        dominant_source = max(distribution.items(), key=lambda x: x[1])
        
        if len(distribution) == 1:
            return f"Single source: {dominant_source[0]}"
        elif dominant_source[1] / sum(distribution.values()) > 0.7:
            return f"Dominant source: {dominant_source[0]}"
        else:
            return "Diverse sources"
