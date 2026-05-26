from typing import List, Dict, Any
from app.vectorstores.vectorstore_manager import vectorstore_manager

class VectorRetriever:
    def __init__(self):
        self.store = vectorstore_manager.get_store()

    def retrieve(self, query: str, k: int = 5, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Performs dense vector retrieval with metadata filtering."""
        return self.store.similarity_search(query, k=k, filter=filter)
