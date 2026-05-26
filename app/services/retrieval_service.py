from typing import List, Dict, Any
from app.agents.retrieval_agent import retrieval_pipeline

class RetrievalService:
    @staticmethod
    def retrieve_chunks(query: str) -> List[Dict[str, Any]]:
        """Direct retrieval helper for analysis panels."""
        chunks, _ = retrieval_pipeline.retrieve(query)
        return chunks
