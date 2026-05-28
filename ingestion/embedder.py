"""Embedding generation with SentenceTransformers and persistent storage."""

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from core.interfaces import IEmbedder
from config.settings import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(IEmbedder):
    """SentenceTransformer-based embedding generator."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedder with specified model.
        
        Args:
            model_name: Model name to use (defaults to settings)
        """
        self.model_name = model_name or settings.embedding.model_name
        # Initialize model with multiple parameters to prevent meta tensor usage
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={
                "low_cpu_mem_usage": False,
                "torch_dtype": "float32"
            }
        )
        logger.info(f"Initialized SentenceTransformer with model: {self.model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        logger.info("Generating embedding for query")
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )
        
        return embedding.tolist()
