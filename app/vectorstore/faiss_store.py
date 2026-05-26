import os
import pickle
import numpy as np
from typing import List, Dict, Any
from sklearn.neighbors import NearestNeighbors
from app.vectorstore.base_vectorstore import BaseVectorStore
from app.embeddings.embedding_factory import embedding_model
from app.utils.logger import logger

class FAISSStore(BaseVectorStore):
    def __init__(self, persist_dir: str = "vector_db/faiss"):
        self.persist_dir = persist_dir
        self.index_file = os.path.join(persist_dir, "faiss_index.pkl")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.documents = []
        self.embeddings = []
        self.nn_model = None
        
        self.load_index()

    def load_index(self):
        """Loads serialized embeddings index from disk if it exists."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.embeddings = data.get("embeddings", [])
                logger.info(f"Loaded FAISS serialized index with {len(self.documents)} docs.")
                self._rebuild_nn_model()
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")

    def save_index(self):
        """Serializes current embeddings and documents to disk."""
        try:
            with open(self.index_file, "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "embeddings": self.embeddings
                }, f)
            logger.info("Saved FAISS serialized index to disk.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _rebuild_nn_model(self):
        if len(self.embeddings) > 0:
            # Fit NearestNeighbors model using cosine metric
            X = np.array(self.embeddings, dtype=np.float32)
            self.nn_model = NearestNeighbors(n_neighbors=min(5, len(X)), metric="cosine")
            self.nn_model.fit(X)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Generates embeddings and appends documents to the index."""
        if not documents:
            return
            
        texts = [doc.get("text", "") for doc in documents]
        logger.info(f"FAISS indexing {len(texts)} documents...")
        
        new_embeddings = embedding_model.embed_documents(texts)
        
        for doc, emb in zip(documents, new_embeddings):
            self.documents.append(doc)
            self.embeddings.append(emb)
            
        self._rebuild_nn_model()
        self.save_index()
        logger.info("FAISS index update complete.")

    def similarity_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Dense similarity search using Scikit-Learn NearestNeighbors and custom metadata filtering."""
        if not self.documents or len(self.embeddings) == 0:
            return []
            
        query_emb = np.array(embedding_model.embed_query(query), dtype=np.float32).reshape(1, -1)
        
        # Apply metadata filtering first (pre-filtering)
        candidate_indices = []
        for idx, doc in enumerate(self.documents):
            meta = doc.get("metadata", {})
            
            # Check filter criteria
            matches = True
            if filter:
                for fk, fv in filter.items():
                    if meta.get(fk) != fv:
                        matches = False
                        break
            if matches:
                candidate_indices.append(idx)
                
        if not candidate_indices:
            return []
            
        # Extract subset of embeddings and documents matching the metadata filter
        filtered_embeddings = np.array([self.embeddings[i] for i in candidate_indices], dtype=np.float32)
        filtered_docs = [self.documents[i] for i in candidate_indices]
        
        # Fit a temporary neighbors model on filtered candidates
        n_neighbors = min(k, len(filtered_docs))
        model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        model.fit(filtered_embeddings)
        
        distances, indices = model.kneighbors(query_emb)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc = filtered_docs[idx]
            similarity = 1.0 - float(dist)
            
            # Format return dict to match base specifications
            results.append({
                "id": str(hash(doc.get("text", ""))),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "score": similarity
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def clear(self) -> None:
        self.documents = []
        self.embeddings = []
        self.nn_model = None
        if os.path.exists(self.index_file):
            try:
                os.remove(self.index_file)
            except Exception as e:
                logger.error(f"Error removing FAISS index file: {e}")
        logger.info("FAISS serialized database cleared.")
