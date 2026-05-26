from typing import List, Dict, Any
from app.vectorstores.base_vectorstore import BaseVectorStore
from app.embeddings.embedding_manager import embedding_manager
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FAISSStore(BaseVectorStore):
    def __init__(self):
        self.docs = []
        self.embeddings = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        texts = [doc["text"] for doc in documents]
        embs = embedding_manager.embed_documents(texts)
        self.docs.extend(documents)
        self.embeddings.extend(embs)

    def similarity_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
            
        query_emb = np.array(embedding_manager.embed_query(query)).reshape(1, -1)
        doc_embs = np.array(self.embeddings)
        
        sims = cosine_similarity(query_emb, doc_embs)[0]
        
        # Sort and filter
        scored_docs = []
        for idx, doc in enumerate(self.docs):
            # Apply filter
            if filter:
                match = True
                for k_f, v_f in filter.items():
                    if doc.get("metadata", {}).get(k_f) != v_f:
                        match = False
                        break
                if not match:
                    continue
                    
            scored_doc = doc.copy()
            scored_doc["score"] = float(sims[idx])
            scored_docs.append(scored_doc)
            
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:k]

    def clear(self) -> None:
        self.docs = []
        self.embeddings = []
