"""Multi-Document Intelligence for cross-document retrieval and source aggregation."""

from .multi_document_intelligence import (
    DocumentSource,
    CrossDocumentResult,
    SourceAggregator,
    CrossDocumentRetriever,
    DocumentIntelligence
)

__all__ = [
    'DocumentSource',
    'CrossDocumentResult',
    'SourceAggregator',
    'CrossDocumentRetriever',
    'DocumentIntelligence'
]
