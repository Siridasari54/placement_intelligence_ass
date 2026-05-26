from typing import List, Dict, Any
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.utils.helpers import compute_rrf
from app.utils.config_loader import config_loader
from app.utils.logger import logger

# Global BM25 retriever instance that can be set after ingestion
_global_bm25_retriever: BM25Retriever = None

def set_global_bm25_retriever(bm25: BM25Retriever) -> None:
    """Set the globally shared BM25Retriever after it has been fitted.
    Allows HybridRetriever to reuse the trained BM25 model.
    """
    global _global_bm25_retriever
    _global_bm25_retriever = bm25

class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever = None):
        self.vector_retriever = VectorRetriever()
        # Use provided BM25, else global if set, else new instance
        if bm25_retriever:
            self.bm25_retriever = bm25_retriever
        elif _global_bm25_retriever:
            self.bm25_retriever = _global_bm25_retriever
        else:
            self.bm25_retriever = BM25Retriever()

    def set_bm25_retriever(self, bm25_retriever: BM25Retriever) -> None:
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, k: int = 5, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Combines Vector and BM25 search results using Reciprocal Rank Fusion."""
        # Config weights
        weight_vec = config_loader.get("retrieval.hybrid.weight_vector", 0.6)
        weight_kw = config_loader.get("retrieval.hybrid.weight_keyword", 0.4)
        
        # Retrieve from both systems
        vector_results = self.vector_retriever.retrieve(query, k=k*2, filter=filter)
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query, k=k*2, filter=filter)
            
        if not vector_results and not bm25_results:
            return []
            
        # Merge using RRF helper
        merged_results = compute_rrf([vector_results, bm25_results], k=60)
        
        logger.info(f"Hybrid search returned {len(merged_results)} merged chunks.")
        return merged_results[:k]
