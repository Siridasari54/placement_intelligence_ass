from typing import List, Dict, Any
from langchain_core.documents import Document
from retrieval.vectorstore.vectorstore_manager import vectorstore_manager

class VectorRetriever:
    def __init__(self):
        self.store = vectorstore_manager.get_store()

    def retrieve(self, query: str, k: int = 5, filter: Dict[str, Any] = None) -> List[Document]:
        """Performs dense vector retrieval with metadata filtering.
        Returns a list of LangChain Document objects for consistency.
        """
        raw_results = self.store.similarity_search(query, k=k, filter=filter)
        # Convert dict results to Document objects
        documents = []
        for res in raw_results:
            # Expect keys 'text' and 'metadata'
            text = res.get("text", "")
            metadata = res.get("metadata", {})
            documents.append(Document(page_content=text, metadata=metadata))
        return documents
