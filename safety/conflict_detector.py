"""Conflict detection between official and portal-scraped data."""

from typing import List, Tuple
from langchain_core.documents import Document
from core.interfaces import ISafetyChecker
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class ConflictDetector(ISafetyChecker):
    """Detects conflicts between official and portal-scraped placement data."""
    
    def __init__(self, threshold: float = 0.8):
        """Initialize the conflict detector.
        
        Args:
            threshold: Similarity threshold for conflict detection
        """
        self.threshold = threshold
        logger.info(f"ConflictDetector initialized with threshold {threshold}")
    
    def check_conflict(self, documents: List[Document]) -> List[Document]:
        """Check for conflicts between documents.
        
        Args:
            documents: List of Document objects to check
            
        Returns:
            List of conflicting Document objects
        """
        logger.info(f"Checking for conflicts in {len(documents)} documents")
        
        conflicts = []
        
        # Group documents by company
        company_docs = {}
        for doc in documents:
            company = doc.metadata.get("company", "unknown")
            if company not in company_docs:
                company_docs[company] = []
            company_docs[company].append(doc)
        
        # Check for conflicts within each company group
        for company, docs in company_docs.items():
            if len(docs) > 1:
                # Check for conflicting sources (official vs portal)
                official_docs = [d for d in docs if d.metadata.get("source") == "official"]
                portal_docs = [d for d in docs if d.metadata.get("source") == "portal"]
                
                if official_docs and portal_docs:
                    # Mark portal documents as potential conflicts
                    conflicts.extend(portal_docs)
        
        logger.info(f"Found {len(conflicts)} conflicting documents")
        return conflicts
    
    def check_out_of_corpus(self, query: str, documents: List[Document]) -> bool:
        """Check if query is out of corpus scope.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            True if query is out of corpus, False otherwise
        """
        logger.info("Checking if query is out of corpus scope")
        
        # Simple heuristic: if no documents retrieved or low relevance
        if not documents:
            return True
        
        # Check if documents contain relevant keywords from query
        query_keywords = set(query.lower().split())
        relevant_docs = 0
        
        for doc in documents:
            doc_text = doc.page_content.lower()
            if any(keyword in doc_text for keyword in query_keywords):
                relevant_docs += 1
        
        # If less than threshold of documents are relevant, consider out of corpus
        relevance_ratio = relevant_docs / len(documents)
        threshold = getattr(settings.safety, 'out_of_corpus_threshold', 0.3)
        is_out_of_corpus = relevance_ratio < threshold
        
        logger.info(f"Relevance ratio: {relevance_ratio:.2f}, Out of corpus: {is_out_of_corpus}")
        return is_out_of_corpus
