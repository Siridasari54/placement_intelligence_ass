from typing import List, Dict, Any
from retrieval.vectorstore.base_vectorstore import BaseVectorStore
import logging

logger = logging.getLogger(__name__)

class QdrantStore(BaseVectorStore):
    def __init__(self):
        logger.warning("Qdrant client not installed. Running in STUB mode.")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        logger.info(f"STUB Qdrant add_documents: {len(documents)} documents.")

    def similarity_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        logger.info(f"STUB Qdrant similarity_search for query: '{query}'")
        return []

    def clear(self) -> None:
        logger.info("STUB Qdrant database cleared.")
