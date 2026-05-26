from typing import List
from app.embeddings.embedding_factory import EmbeddingFactory
from app.embeddings.base_embedding import BaseEmbedding

class EmbeddingManager:
    _instance = None
    _model: BaseEmbedding = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._model = EmbeddingFactory.get_embedding_model()
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

embedding_manager = EmbeddingManager()
