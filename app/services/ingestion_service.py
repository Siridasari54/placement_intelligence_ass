import os
from typing import Dict, Any, List

from app.utils.logger import logger
from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.chunking.chunk_manager import ChunkManager
from app.vectorstore.vectorstore_manager import vectorstore_manager
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import set_global_bm25_retriever


class IngestionService:
    def __init__(self):
        self.pipeline = IngestionPipeline()
        self.chunk_manager = ChunkManager()
        self.bm25_retriever = BM25Retriever()

    def ingest_raw_text(self, raw_text: str) -> Dict[str, Any]:
        """Ingest raw text dataset, create chunks, fit BM25, and index in Chroma."""
        logger.info("Starting ingestion workflow...")

        # 1. Parse raw text into structured JSON tables
        parsed_data = self.pipeline.process_raw_dataset_text(raw_text)

        # 2. Chunk structured data
        chunks = self.chunk_manager.chunk_all_data()

        # 3. Fit BM25 and expose globally for hybrid retrieval
        self.bm25_retriever.fit(chunks)
        set_global_bm25_retriever(self.bm25_retriever)

        # 4. Clear old and index in Chroma Vector DB
        store = vectorstore_manager.get_store()
        try:
            store.clear()
        except Exception as e:
            logger.warning(f"Failed to clear vector store (might be new): {e}")

        store.add_documents(chunks)

        logger.info("Ingestion and indexing pipeline completed successfully.")
        return {
            "status": "success",
            "chunks_count": len(chunks),
            "companies_count": len(parsed_data.get("eligibility", []))
        }

    def get_bm25_retriever(self) -> BM25Retriever:
        return self.bm25_retriever


ingestion_service = IngestionService()
