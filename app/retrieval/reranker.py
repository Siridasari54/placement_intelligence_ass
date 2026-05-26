from typing import List, Dict, Any
from app.utils.config_loader import config_loader
from app.utils.logger import logger

class Reranker:
    def __init__(self):
        self.enabled = config_loader.get("retrieval.reranker.enabled", True)
        self.model_name = config_loader.get("retrieval.reranker.model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_n = config_loader.get("retrieval.reranker.top_n", 5)
        self.model = None
        
        if self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder reranker: {self.model_name}")
                self.model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder model {self.model_name}: {e}. Reranking will fall back to local scores.")

    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranks retrieved candidate chunks against the search query."""
        if not chunks or not self.enabled:
            return chunks
            
        if not self.model:
            # Fall back: sort by original retrieval scores
            return sorted(chunks, key=lambda x: x.get("score", x.get("rrf_score", 0.0)), reverse=True)[:self.top_n]
            
        try:
            # Prepare pairs: (query, document_text)
            pairs = [[query, chunk.get("text", "")] for chunk in chunks]
            scores = self.model.predict(pairs)
            
            # Update scores in chunks
            reranked_chunks = []
            for idx, score in enumerate(scores):
                chunk = chunks[idx].copy()
                chunk["rerank_score"] = float(score)
                reranked_chunks.append(chunk)
                
            # Sort by rerank score descending
            reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info(f"Reranking complete. Top score: {reranked_chunks[0]['rerank_score']:.4f}")
            return reranked_chunks[:self.top_n]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fall back to original order
            return chunks[:self.top_n]
