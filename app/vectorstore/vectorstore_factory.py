import os
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.faiss_store import FAISSStore
from app.vectorstore.pinecone_store import PineconeStore
from app.vectorstore.qdrant_store import QdrantStore
from app.utils.config_loader import config_loader
from app.utils.logger import logger

class VectorStoreFactory:
    _instances = {}

    @classmethod
    def get_store(cls, provider: str = None):
        if provider is None:
            # Check constants/config
            provider = config_loader.get("retrieval.vectorstore.provider", "chroma")
        
        provider = provider.lower()
        if provider not in cls._instances:
            if provider == "chroma":
                persist_path = config_loader.get("retrieval.chroma.persist_dir", "vector_db/chroma")
                cls._instances[provider] = ChromaStore(persist_dir=persist_path)
            elif provider == "faiss":
                persist_path = config_loader.get("retrieval.faiss.persist_dir", "vector_db/faiss")
                cls._instances[provider] = FAISSStore(persist_dir=persist_path)
            elif provider == "pinecone":
                cls._instances[provider] = PineconeStore()
            elif provider == "qdrant":
                cls._instances[provider] = QdrantStore()
            else:
                raise ValueError(f"Unsupported Vector Store provider: {provider}")
                
        return cls._instances[provider]
