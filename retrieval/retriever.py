"""Hybrid retriever combining dense vector and sparse BM25 retrieval."""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from core.interfaces import IRetriever, IVectorStore, IEmbedder
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)


class HybridRetriever(IRetriever):
    """Hybrid retriever combining dense vector and sparse BM25 retrieval with RRF."""
    
    def __init__(
        self,
        vector_store: IVectorStore,
        embedder: IEmbedder,
        corpus: List[str] = None
    ):
        """Initialize the hybrid retriever.
        
        Args:
            vector_store: Vector storage for dense retrieval
            embedder: Embedding generator
            corpus: List of corpus texts for BM25 indexing
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = None
        
        if corpus:
            self._index_bm25(corpus)
        
        logger.info("HybridRetriever initialized")
    
    def _index_bm25(self, corpus: List[str]) -> None:
        """Index corpus for BM25 retrieval.
        
        Args:
            corpus: List of corpus texts
        """
        logger.info("Indexing corpus for BM25")
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 indexing complete")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved Document objects
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Dense retrieval
        dense_results = self.vector_store.similarity_search(query, k=k)
        
        # Sparse retrieval (BM25)
        sparse_results = []
        if self.bm25:
            tokenized_query = query.split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
            # Get top-k sparse results
            top_indices = sparse_scores.argsort()[-k:][::-1]
            sparse_results = [dense_results[i] for i in top_indices if i < len(dense_results)]
        
        # Combine using Reciprocal Rank Fusion (RRF)
        combined = self._rrf_fusion(dense_results, sparse_results, k)
        
        logger.info(f"Retrieved {len(combined)} documents")
        return combined
    
    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int,
        k_rrf: int = 60
    ) -> List[Document]:
        """Combine results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results
            k: Number of final results
            k_rrf: RRF constant
            
        Returns:
            Combined and reranked Document objects
        """
        scores = {}
        
        # Score dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc.get("text", "")
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
        
        # Score sparse results
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.get("text", "")
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
        
        # Sort by score and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Convert back to Document objects
        documents = []
        seen = set()
        for text, score in sorted_results:
            if text not in seen:
                # Find the original document
                for doc in dense_results:
                    if doc.get("text", "") == text:
                        documents.append(Document(
                            page_content=text,
                            metadata={**doc.get("metadata", {}), "rrf_score": score}
                        ))
                        seen.add(text)
                        break
        
        return documents
