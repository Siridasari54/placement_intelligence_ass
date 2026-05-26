import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from app.utils.logger import logger

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def fit(self, chunks: List[Dict[str, Any]]) -> None:
        """Fits the BM25 model on a list of document chunks."""
        if not chunks:
            logger.warning("Empty chunk list provided for BM25 fitting.")
            return
            
        self.chunks = chunks
        
        # Tokenize corpus
        corpus_tokens = [self._tokenize(chunk.get("text", "")) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Successfully fitted BM25 model on {len(chunks)} chunks.")

    def retrieve(self, query: str, k: int = 5, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Performs sparse keyword retrieval with optional metadata pre-filtering."""
        if not self.bm25 or not self.chunks:
            return []
            
        # Apply pre-filtering if filter is provided
        filtered_indices = []
        filtered_chunks = []
        
        for idx, chunk in enumerate(self.chunks):
            if filter:
                match = True
                for k_f, v_f in filter.items():
                    if chunk.get("metadata", {}).get(k_f) != v_f:
                        match = False
                        break
                if not match:
                    continue
            filtered_indices.append(idx)
            filtered_chunks.append(chunk)
            
        if not filtered_chunks:
            return []
            
        # Re-initialize BM25 for the filtered subset if a filter was applied
        if filter:
            sub_tokens = [self._tokenize(c.get("text", "")) for c in filtered_chunks]
            sub_bm25 = BM25Okapi(sub_tokens)
            query_tokens = self._tokenize(query)
            scores = sub_bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for t_idx in top_indices:
                if scores[t_idx] > 0:
                    item = filtered_chunks[t_idx].copy()
                    item["score"] = float(scores[t_idx])
                    results.append(item)
            return results
        else:
            query_tokens = self._tokenize(query)
            scores = self.bm25.get_scores(query_tokens)
            
            # Sort all indices by score
            import numpy as np
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for t_idx in top_indices:
                if scores[t_idx] > 0:
                    item = self.chunks[t_idx].copy()
                    item["score"] = float(scores[t_idx])
                    results.append(item)
            return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer splitting by alphanumeric tokens."""
        return re.findall(r"\b\w+\b", text.lower())
