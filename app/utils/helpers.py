import uuid
from typing import List, Dict, Any

def generate_id() -> str:
    return str(uuid.uuid4())

def compute_rrf(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Computes Reciprocal Rank Fusion (RRF) scores across multiple search result lists.
    RRF score is calculated as sum(1 / (rank + k)) for each unique document.
    """
    rrf_scores = {}
    doc_map = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            # Use text or id as key
            text = doc.get("text", "")
            if not text:
                continue
                
            doc_hash = str(hash(text))
            doc_map[doc_hash] = doc
            
            if doc_hash not in rrf_scores:
                rrf_scores[doc_hash] = 0.0
                
            # rank is 0-indexed, so we add 1
            rrf_scores[doc_hash] += 1.0 / ((rank + 1) + k)

    # Sort documents by their RRF score descending
    sorted_hashes = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    final_results = []
    for dh in sorted_hashes:
        doc_copy = doc_map[dh].copy()
        doc_copy["score"] = float(rrf_scores[dh])
        final_results.append(doc_copy)
        
    return final_results