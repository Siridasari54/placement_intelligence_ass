"""In-memory vector store for development and testing."""

from typing import List, Dict, Any
import numpy as np
from retrieval.vectorstore.base_vectorstore import BaseVectorStore
import logging

logger = logging.getLogger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store for development and testing."""
    
    def __init__(self):
        """Initialize the in-memory vector store."""
        self.documents = []
        self.embeddings = []
        logger.info("InMemoryVectorStore initialized")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
        """
        logger.info(f"Adding {len(documents)} documents to in-memory vector store")
        for doc in documents:
            self.documents.append(doc)
        logger.info(f"Total documents in store: {len(self.documents)}")
    
    def similarity_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        logger.info(f"Searching for {k} similar documents for query: {query}")
        
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        # Simple keyword-based search for now
        # In production, this would use actual vector similarity
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.documents:
            text = doc.get('text', '').lower()
            metadata = doc.get('metadata', {})
            
            # Apply filter if provided
            if filter:
                filter_match = True
                for key, value in filter.items():
                    if metadata.get(key) != value:
                        filter_match = False
                        break
                if not filter_match:
                    continue
            
            # Simple keyword matching score
            query_words = set(query_lower.split())
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            score = overlap / len(query_words) if query_words else 0
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        results = [doc for doc, score in scored_docs[:k]]
        
        logger.info(f"Returning {len(results)} documents")
        return results
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        logger.info("Clearing in-memory vector store")
        self.documents = []
        self.embeddings = []
