from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Splits text and copies metadata to each chunk."""
        docs = self.splitter.create_documents([text], metadatas=[metadata or {}])
        return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
