from typing import List, Dict, Any
from app.vectorstore.base_vectorstore import BaseVectorStore
from app.utils.logger import logger

class PineconeStore(BaseVectorStore):
    def __init__(self):
        logger.warning("Pinecone client not installed. Running in STUB mode.")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        logger.info(f"STUB Pinecone add_documents: {len(documents)} documents.")

    def similarity_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        logger.info(f"STUB Pinecone similarity_search for query: '{query}'")
        return []

    def clear(self) -> None:
        logger.info("STUB Pinecone database cleared.")
