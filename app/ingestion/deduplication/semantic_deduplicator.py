from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.utils.logger import logger

class SemanticDeduplicator:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def deduplicate(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates a list of text chunks based on semantic similarity."""
        if not chunks:
            return []
            
        logger.info(f"Deduplicating {len(chunks)} chunks using threshold={self.threshold}")
        
        # We separate chunks by text content
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Check exact duplicates first
        unique_indices = []
        seen_texts = set()
        
        for idx, text in enumerate(texts):
            # Clean text minimal for exact match check
            normalized = " ".join(text.lower().split())
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_indices.append(idx)
                
        exact_dedup_chunks = [chunks[i] for i in unique_indices]
        logger.info(f"After exact deduplication: {len(exact_dedup_chunks)} chunks remain")
        
        if len(exact_dedup_chunks) <= 1:
            return exact_dedup_chunks
            
        # Perform TF-IDF Cosine Similarity for near-duplicates
        final_chunks = []
        final_texts = []
        
        for chunk in exact_dedup_chunks:
            text = chunk.get("text", "")
            if not final_texts:
                final_chunks.append(chunk)
                final_texts.append(text)
                continue
                
            # Compute TF-IDF of final_texts + current text
            vectorizer = TfidfVectorizer(min_df=1)
            tfidf = vectorizer.fit_transform(final_texts + [text])
            
            # Compare current text (last element) with all preceding ones
            sims = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
            
            if np.max(sims) < self.threshold:
                final_chunks.append(chunk)
                final_texts.append(text)
            else:
                dup_idx = np.argmax(sims)
                logger.debug(f"Detected duplicate block: '{text[:40]}...' matches '{final_texts[dup_idx][:40]}...' with similarity {sims[dup_idx]:.2f}")
                
        logger.info(f"After semantic deduplication: {len(final_chunks)} chunks remain")
        return final_chunks
