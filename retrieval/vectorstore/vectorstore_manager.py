from retrieval.vectorstore.vectorstore_factory import VectorStoreFactory
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            # Default to Chroma store, can be switched based on config
            cls._instance._store = VectorStoreFactory.get_store("chroma")
        return cls._instance

    def get_store(self, provider: str = None):
        """Returns the active vector store instance."""
        if provider:
            self._store = VectorStoreFactory.get_store(provider)
        return self._store

    def build(self, docs, embedding_model=None):
        """Backwards compatibility: build and index vector store."""
        logger.info("VectorStoreManager building and indexing documents...")
        self._store.clear()
        self._store.add_documents(docs)
        return self._store

    def load(self):
        """Backwards compatibility: load and return vector store."""
        return self._store

# Standard shared singleton instance used in older import styles
vectorstore_manager = VectorStoreManager()
