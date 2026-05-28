import os
from retrieval.vectorstore.chroma_store import ChromaStore
from retrieval.vectorstore.faiss_store import FAISSStore
from retrieval.vectorstore.pinecone_store import PineconeStore
from retrieval.vectorstore.qdrant_store import QdrantStore
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    _instances = {}

    @classmethod
    def get_store(cls, provider: str = None):
        if provider is None:
            # Check constants/config
            provider = "chroma"  # Default to chroma, can be extended to read from settings
        
        provider = provider.lower()
        if provider not in cls._instances:
            if provider == "chroma":
                persist_path = "vector_db/chroma"  # Default path, can be extended to read from settings
                cls._instances[provider] = ChromaStore(persist_dir=persist_path)
            elif provider == "faiss":
                persist_path = "vector_db/faiss"  # Default path, can be extended to read from settings
                cls._instances[provider] = FAISSStore(persist_dir=persist_path)
            elif provider == "pinecone":
                cls._instances[provider] = PineconeStore()
            elif provider == "qdrant":
                cls._instances[provider] = QdrantStore()
            else:
                raise ValueError(f"Unsupported Vector Store provider: {provider}")
                
        return cls._instances[provider]
