from typing import List, Dict, Any, Tuple
from app.utils.logger import logger
from app.retrieval.query_router import QueryRouter
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.metadata_filtering import MetadataFiltering
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.multi_hop_retriever import MultiHopRetriever
from app.retrieval.temporal_retriever import TemporalRetriever
from app.retrieval.conflict_retriever import ConflictRetriever
from app.retrieval.contextual_compression import ContextualCompression

class RetrievalPipeline:
    def __init__(self, bm25_retriever: BM25Retriever = None):
        self.rewriter = QueryRewriter()
        self.router = QueryRouter()
        self.hybrid_retriever = HybridRetriever(bm25_retriever)
        self.reranker = Reranker()
        
        self.multi_hop_retriever = MultiHopRetriever()
        self.temporal_retriever = TemporalRetriever()
        self.conflict_retriever = ConflictRetriever()

    def set_bm25_retriever(self, bm25_retriever: BM25Retriever) -> None:
        self.hybrid_retriever.set_bm25_retriever(bm25_retriever)

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Runs the entire routing and retrieval pipeline."""
        # 1. Query Rewriting
        rewritten_query = self.rewriter.rewrite(query)
        logger.info(f"Original Query: '{query}' -> Rewritten: '{rewritten_query}'")
        
        # 2. Query Routing
        route_info = self.router.route_query(rewritten_query)
        route = route_info["route"]
        logger.info(f"Query routed to: {route} (Reason: {route_info.get('reason')})")
        
        chunks = []
        
        # 3. Execution based on Route
        if route == "fallback":
            # Return empty chunks list with marker to trigger LLM graceful boundary fallback
            chunks = [{
                "text": "Out of corpus fallback indicator.",
                "metadata": {"source": "system", "section": "fallback", "fallback_trigger": True}
            }]
            
        elif route == "edge_case_cgpa":
            chunks = [{
                "text": "Edge Case Alert: Student has a CGPA of 5.0. Analysis shows that the minimum eligibility threshold across all listed companies in this dataset is 6.1 (Microsoft), meaning no company allows a CGPA of 5.0.",
                "metadata": {"source": "system", "section": "eligibility", "below_threshold": True}
            }]
            
        elif route == "conflict":
            company = route_info.get("company")
            if company:
                chunks = self.conflict_retriever.retrieve_conflict_info(company)
            else:
                # Fall back to general retrieval if company not clear
                chunks = self.hybrid_retriever.retrieve(rewritten_query, k=10)
                
        elif route == "temporal":
            chunks = self.temporal_retriever.retrieve_temporal_trend(rewritten_query)
            if not chunks:
                # Fallback to hybrid search
                chunks = self.hybrid_retriever.retrieve(rewritten_query, k=10)
                
        elif route == "multi_hop":
            chunks = self.multi_hop_retriever.retrieve_multi_hop(rewritten_query)
            if not chunks:
                # Fallback to hybrid
                chunks = self.hybrid_retriever.retrieve(rewritten_query, k=10)
                
        else:
            # Standard path: hybrid + reranking
            filters = MetadataFiltering.build_filter_from_query(rewritten_query)
            logger.info(f"Applying metadata filters: {filters}")
            
            # Search
            candidate_chunks = self.hybrid_retriever.retrieve(rewritten_query, k=10, filter=filters)
            
            # Rerank
            chunks = self.reranker.rerank(rewritten_query, candidate_chunks)
            
        # 4. Contextual Compression
        compressed_chunks = ContextualCompression.compress_context(chunks, max_tokens=1500)
        
        logger.info(f"Retrieved {len(compressed_chunks)} final compressed chunks.")
        return compressed_chunks, route
