import os
from typing import List, Dict, Any
import chromadb
from app.vectorstore.base_vectorstore import BaseVectorStore
from app.embeddings.embedding_factory import embedding_model
from app.utils.logger import logger

class ChromaStore(BaseVectorStore):
    def __init__(self, persist_dir: str = "vector_db/chroma"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"Initializing persistent ChromaDB client at: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="placement_data",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Adds a list of document chunks to Chroma."""
        if not documents:
            return
            
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.get("id") or str(hash(doc.get("text", ""))))
            texts.append(doc.get("text", ""))
            
            # Sanitize metadata (Chroma requires dict values to be str, int, float, bool)
            meta = doc.get("metadata", {}).copy()
            sanitized_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                else:
                    sanitized_meta[k] = str(v)
            metadatas.append(sanitized_meta)
            
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = embedding_model.embed_documents(texts)
        
        logger.info(f"Writing {len(texts)} documents to Chroma collection...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info("Chroma indexing complete.")

    def similarity_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Queries Chroma database using text embeddings, applying optional filters."""
        query_embedding = embedding_model.embed_query(query)
        
        # Format filter for Chroma where clause
        where_clause = None
        if filter:
            # Clean filters to strings/numbers/bools for Chroma compatibility
            cleaned_filter = {}
            for k_filt, v_filt in filter.items():
                if isinstance(v_filt, (str, int, float, bool)):
                    cleaned_filter[k_filt] = v_filt
                    
            if len(cleaned_filter) == 1:
                where_clause = cleaned_filter
            elif len(cleaned_filter) > 1:
                where_clause = {"$and": [{k_f: v_f} for k_f, v_f in cleaned_filter.items()]}
                
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )
        
        docs = []
        if results and "documents" in results and results["documents"]:
            for idx in range(len(results["documents"][0])):
                doc_text = results["documents"][0][idx]
                doc_meta = results["metadatas"][0][idx]
                doc_id = results["ids"][0][idx]
                distance = results["distances"][0][idx] if "distances" in results else 1.0
                similarity = 1.0 - distance
                
                docs.append({
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": doc_meta,
                    "score": float(similarity)
                })
        return docs

    def clear(self) -> None:
        """Deletes the entire Chroma collection."""
        try:
            self.client.delete_collection("placement_data")
            self.collection = self.client.get_or_create_collection("placement_data")
            logger.info("Chroma collection cleared.")
        except Exception as e:
            logger.error(f"Error clearing Chroma database: {e}")
