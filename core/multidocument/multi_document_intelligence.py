"""Multi-Document Intelligence for cross-document retrieval and source aggregation."""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class DocumentSource:
    """Information about a document source."""
    source_id: str
    source_name: str
    document_type: str  # pdf, excel, notice, report, etc.
    total_chunks: int = 0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossDocumentResult:
    """Result of cross-document retrieval."""
    query: str
    sources: List[DocumentSource] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    aggregated_answer: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SourceAggregator:
    """Aggregates information from multiple document sources."""
    
    def __init__(self):
        """Initialize the source aggregator."""
        self.source_registry: Dict[str, DocumentSource] = {}
        
        logger.info("SourceAggregator initialized")
    
    def register_source(self, source: DocumentSource) -> None:
        """Register a document source.
        
        Args:
            source: DocumentSource to register
        """
        self.source_registry[source.source_id] = source
        logger.info(f"Registered source: {source.source_name}")
    
    def aggregate_by_source(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Aggregate documents by their source.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary mapping source IDs to document lists
        """
        source_groups = defaultdict(list)
        
        for doc in documents:
            source_id = doc.metadata.get("source_id", "unknown")
            source_groups[source_id].append(doc)
        
        logger.info(f"Aggregated {len(documents)} documents into {len(source_groups)} sources")
        return dict(source_groups)
    
    def calculate_source_relevance(self, documents: List[Document]) -> Dict[str, float]:
        """Calculate relevance score for each source.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary mapping source IDs to relevance scores
        """
        source_scores = defaultdict(float)
        source_counts = defaultdict(int)
        
        for doc in documents:
            source_id = doc.metadata.get("source_id", "unknown")
            score = doc.metadata.get("score", 0.0)
            source_scores[source_id] += score
            source_counts[source_id] += 1
        
        # Normalize by count
        for source_id in source_scores:
            if source_counts[source_id] > 0:
                source_scores[source_id] /= source_counts[source_id]
        
        return dict(source_scores)
    
    def get_source_summary(self, source_id: str) -> Optional[DocumentSource]:
        """Get summary of a source.
        
        Args:
            source_id: Source ID
            
        Returns:
            DocumentSource if found, None otherwise
        """
        return self.source_registry.get(source_id)


class CrossDocumentRetriever:
    """Retrieves and aggregates information across multiple documents."""
    
    def __init__(self, source_aggregator: SourceAggregator):
        """Initialize the cross-document retriever.
        
        Args:
            source_aggregator: SourceAggregator instance
        """
        self.source_aggregator = source_aggregator
        
        logger.info("CrossDocumentRetriever initialized")
    
    def retrieve_cross_document(
        self,
        query: str,
        retrievers: Dict[str, Any],
        k: int = 5,
        source_weights: Dict[str, float] = None
    ) -> CrossDocumentResult:
        """Retrieve documents across multiple sources.
        
        Args:
            query: User query
            retrievers: Dictionary mapping source IDs to retriever instances
            k: Number of documents to retrieve per source
            source_weights: Optional weights for sources
            
        Returns:
            CrossDocumentResult
        """
        logger.info(f"Cross-document retrieval for query: {query}")
        
        all_documents = []
        source_distribution = defaultdict(int)
        
        # Retrieve from each source
        for source_id, retriever in retrievers.items():
            try:
                docs = retriever.retrieve(query, k=k)
                all_documents.extend(docs)
                source_distribution[source_id] = len(docs)
                
                # Add source metadata to documents
                for doc in docs:
                    doc.metadata["source_id"] = source_id
                    source_info = self.source_aggregator.get_source_summary(source_id)
                    if source_info:
                        doc.metadata["source_name"] = source_info.source_name
                        doc.metadata["document_type"] = source_info.document_type
                
                logger.info(f"Retrieved {len(docs)} documents from source: {source_id}")
            except Exception as e:
                logger.error(f"Failed to retrieve from source {source_id}: {e}")
        
        # Calculate source relevance
        source_relevance = self.source_aggregator.calculate_source_relevance(all_documents)
        
        # Build source information
        sources = []
        for source_id, relevance in source_relevance.items():
            source_info = self.source_aggregator.get_source_summary(source_id)
            if source_info:
                source_info.relevance_score = relevance
                source_info.total_chunks = source_distribution.get(source_id, 0)
                sources.append(source_info)
        
        # Sort sources by relevance
        sources.sort(key=lambda x: x.relevance_score, reverse=True)
        
        result = CrossDocumentResult(
            query=query,
            sources=sources,
            documents=all_documents,
            source_distribution=dict(source_distribution),
            confidence=sum(source_relevance.values()) / len(source_relevance) if source_relevance else 0.0
        )
        
        logger.info(f"Cross-document retrieval complete: {len(all_documents)} documents from {len(sources)} sources")
        return result
    
    def merge_cross_source_answers(
        self,
        cross_result: CrossDocumentResult,
        generator
    ) -> str:
        """Merge answers from multiple sources.
        
        Args:
            cross_result: Cross-document retrieval result
            generator: Generator instance
            
        Returns:
            Aggregated answer
        """
        logger.info("Merging cross-source answers")
        
        if not cross_result.documents:
            return "No information found across documents."
        
        # Group documents by source
        source_groups = self.source_aggregator.aggregate_by_source(cross_result.documents)
        
        # Generate answer for each source
        source_answers = []
        for source_id, docs in source_groups.items():
            source_info = self.source_aggregator.get_source_summary(source_id)
            source_name = source_info.source_name if source_info else source_id
            
            # Generate answer for this source
            source_answer = generator.generate(cross_result.query, docs)
            source_answers.append({
                "source": source_name,
                "answer": source_answer,
                "doc_count": len(docs)
            })
        
        # Aggregate answers
        aggregated = self._aggregate_answers(source_answers, cross_result.sources)
        
        cross_result.aggregated_answer = aggregated
        return aggregated
    
    def _aggregate_answers(
        self,
        source_answers: List[Dict[str, Any]],
        sources: List[DocumentSource]
    ) -> str:
        """Aggregate answers from multiple sources.
        
        Args:
            source_answers: List of source answers
            sources: List of document sources
            
        Returns:
            Aggregated answer
        """
        if not source_answers:
            return "No information available."
        
        # Build aggregated response
        parts = []
        
        # Add overview
        parts.append(f"Based on information from {len(source_answers)} document sources:")
        
        # Add each source's contribution
        for answer in source_answers:
            source_name = answer["source"]
            answer_text = answer["answer"]
            doc_count = answer["doc_count"]
            
            parts.append(f"\n**From {source_name}** ({doc_count} documents):")
            parts.append(answer_text)
        
        # Add synthesis
        if len(source_answers) > 1:
            parts.append("\n**Synthesis:**")
            parts.append("The above information is aggregated from multiple document sources. "
                        "Each source provides relevant context for the query.")
        
        return "\n".join(parts)


class DocumentIntelligence:
    """Unified multi-document intelligence system."""
    
    def __init__(self):
        """Initialize the document intelligence system."""
        self.source_aggregator = SourceAggregator()
        self.cross_document_retriever = CrossDocumentRetriever(self.source_aggregator)
        
        logger.info("DocumentIntelligence initialized")
    
    def register_document_source(
        self,
        source_id: str,
        source_name: str,
        document_type: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Register a document source.
        
        Args:
            source_id: Unique source ID
            source_name: Human-readable source name
            document_type: Type of documents (pdf, excel, etc.)
            metadata: Optional metadata
        """
        source = DocumentSource(
            source_id=source_id,
            source_name=source_name,
            document_type=document_type,
            metadata=metadata or {}
        )
        self.source_aggregator.register_source(source)
    
    def retrieve_across_documents(
        self,
        query: str,
        retrievers: Dict[str, Any],
        k: int = 5,
        merge_answers: bool = True,
        generator = None
    ) -> CrossDocumentResult:
        """Retrieve and optionally merge answers across documents.
        
        Args:
            query: User query
            retrievers: Dictionary mapping source IDs to retrievers
            k: Documents per source
            merge_answers: Whether to merge answers
            generator: Optional generator for answer merging
            
        Returns:
            CrossDocumentResult
        """
        # Retrieve across documents
        result = self.cross_document_retriever.retrieve_cross_document(
            query, retrievers, k=k
        )
        
        # Merge answers if requested
        if merge_answers and generator:
            aggregated = self.cross_document_retriever.merge_cross_source_answers(
                result, generator
            )
            result.aggregated_answer = aggregated
        
        return result
    
    def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered sources.
        
        Returns:
            Dictionary of source statistics
        """
        sources = list(self.source_aggregator.source_registry.values())
        
        return {
            "total_sources": len(sources),
            "source_types": self._count_by_type(sources),
            "sources": [
                {
                    "source_id": s.source_id,
                    "source_name": s.source_name,
                    "document_type": s.document_type,
                    "total_chunks": s.total_chunks,
                    "relevance_score": s.relevance_score
                }
                for s in sources
            ]
        }
    
    def _count_by_type(self, sources: List[DocumentSource]) -> Dict[str, int]:
        """Count sources by document type.
        
        Args:
            sources: List of document sources
            
        Returns:
            Dictionary mapping types to counts
        """
        type_counts = defaultdict(int)
        for source in sources:
            type_counts[source.document_type] += 1
        return dict(type_counts)
