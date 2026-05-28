"""Configuration management for the Placement Intelligence Assistant."""

import os
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100


@dataclass
class GenerationConfig:
    """Configuration for answer generation."""
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    max_tokens: int = 1024
    context_window: int = 8192


@dataclass
class SafetyConfig:
    """Configuration for safety layer."""
    conflict_threshold: float = 0.8
    out_of_corpus_threshold: float = 0.3
    enable_conflict_detection: bool = True
    enable_fallback_guard: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation system."""
    evaluation_queries_file: str = "evaluation/queries.py"
    metrics_output_dir: str = "evaluation/results"


@dataclass
class DataConfig:
    """Configuration for data paths."""
    processed_dir: str = "data/processed"
    raw_dir: str = "data/raw"
    index_dir: str = "data/indexes"


@dataclass
class UIConfig:
    """Configuration for UI settings."""
    title: str = "Placement Intelligence Assistant"
    theme: str = "dark"
    max_chat_history: int = 50


@dataclass
class Settings:
    """Main settings class containing all configuration."""
    
    # API Keys
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    
    # Module configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # System settings
    # Optional ChromaDB tenant and database for future multitenancy
    chromadb_tenant: Optional[str] = None
    chromadb_database: Optional[str] = None
    debug: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Create necessary directories
        os.makedirs(self.data.processed_dir, exist_ok=True)
        os.makedirs(self.data.raw_dir, exist_ok=True)
        os.makedirs(self.data.index_dir, exist_ok=True)
        os.makedirs(self.evaluation.metrics_output_dir, exist_ok=True)


# Global settings instance
settings = Settings()
