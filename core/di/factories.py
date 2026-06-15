"""Service factories for creating service instances with proper dependency injection."""

from typing import Dict, Any, Optional, Tuple
from core.di.container import ServiceContainer
from core.interfaces import (
    IParser, IChunker, IEmbedder, IVectorStore,
    IRetriever, IReranker, IRefiner, IGenerator,
    ISafetyChecker
)
from ingestion.parser import MultimodalParser
from ingestion.chunker import IntelligentChunker
from ingestion.embedder import SentenceTransformerEmbedder
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.vectorstore.in_memory_store import InMemoryVectorStore
from generation.refiner import ContextRefiner
from generation.generator import GroqGenerator
from config.settings import settings

# New engineering services
from core.query_planning.query_planner import QueryPlanner
from core.observability.pipeline_tracer import PipelineTracer
from core.memory.memory_system import AIMemorySystem
from core.analytics.retrieval_analytics import RetrievalAnalytics
from core.reliability.system_reliability import SystemReliabilityLayer
from core.retrieval.adaptive_strategy import AdaptiveRetrievalStrategy, DynamicRetriever
from core.pipeline import RAGPipeline
from safety.conflict_detector import ConflictDetector
from safety.fallback_guard import FallbackGuard
import logging

# Multi-hop retriever (optional)
MultiHopRetriever = None
try:
    from retrieval.multi_hop_retriever import MultiHopRetriever
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory for creating service instances with dependency injection."""
    
    def __init__(self, container: ServiceContainer):
        """Initialize the service factory.
        
        Args:
            container: Service container instance
        """
        self.container = container
        logger.info("ServiceFactory initialized")
    
    def create_parser(self) -> IParser:
        """Create parser instance.
        
        Returns:
            Parser instance
        """
        return MultimodalParser()
    
    def create_chunker(self) -> IChunker:
        """Create intelligent chunker instance with metadata enrichment.
        
        Returns:
            Chunker instance
        """
        return IntelligentChunker()
    
    def create_embedder(self) -> IEmbedder:
        """Create embedder instance.
        
        Returns:
            Embedder instance
        """
        return SentenceTransformerEmbedder(
            model_name=settings.embedding.model_name
        )
    
    def create_vector_store(self) -> IVectorStore:
        """Create vector store instance.
        
        Returns:
            Vector store instance
        """
        from retrieval.vectorstore.vectorstore_factory import VectorStoreFactory
        # Use FAISS as default vector store (more stable than ChromaDB)
        return VectorStoreFactory.get_store("faiss")
    
    def create_retriever(self) -> IRetriever:
        """Create retriever instance with dependencies.
        
        Returns:
            Retriever instance
        """
        vector_store = self.container.get_service(IVectorStore)
        embedder = self.container.get_service(IEmbedder)
        
        return HybridRetriever(
            vector_store=vector_store,
            embedder=embedder
        )
    
    def create_reranker(self) -> IReranker:
        """Create enhanced reranker instance with thresholding and deduplication.
        
        Returns:
            Reranker instance
        """
        return CrossEncoderReranker(
            model_name=settings.retrieval.rerank_model,
            relevance_threshold=0.3,
            deduplicate=True
        )
    
    def create_refiner(self) -> IRefiner:
        """Create refiner instance.
        
        Returns:
            Refiner instance
        """
        return ContextRefiner()
    
    def create_generator(self) -> IGenerator:
        """Create generator instance.
        
        Returns:
            Generator instance
        """
        return GroqGenerator(
            api_key=settings.groq_api_key
        )
    
    def create_query_planner(self) -> QueryPlanner:
        """Create query planner instance.
        
        Returns:
            QueryPlanner instance
        """
        adaptive_strategy = self.container.get_service(AdaptiveRetrievalStrategy)
        return QueryPlanner(adaptive_strategy=adaptive_strategy)
    
    def create_pipeline_tracer(self) -> PipelineTracer:
        """Create pipeline tracer instance.
        
        Returns:
            PipelineTracer instance
        """
        return PipelineTracer()
    
    def create_memory_system(self) -> AIMemorySystem:
        """Create memory system instance.
        
        Returns:
            AIMemorySystem instance
        """
        return AIMemorySystem()
    
    def create_retrieval_analytics(self) -> RetrievalAnalytics:
        """Create retrieval analytics instance.
        
        Returns:
            RetrievalAnalytics instance
        """
        return RetrievalAnalytics()
    
    def create_reliability_layer(self) -> SystemReliabilityLayer:
        """Create reliability layer instance.
        
        Returns:
            SystemReliabilityLayer instance
        """
        return SystemReliabilityLayer()
    
    def create_adaptive_strategy(self) -> AdaptiveRetrievalStrategy:
        """Create adaptive retrieval strategy instance.
        
        Returns:
            AdaptiveRetrievalStrategy instance
        """
        return AdaptiveRetrievalStrategy()
    
    def create_dynamic_retriever(self) -> DynamicRetriever:
        """Create dynamic retriever instance.
        
        Returns:
            DynamicRetriever instance
        """
        adaptive_strategy = self.container.get_service(AdaptiveRetrievalStrategy)
        return DynamicRetriever(adaptive_strategy)
    

    
    def create_multi_hop_retriever(self):
        """Create multi-hop retriever instance with query rewriting.
        
        Returns:
            MultiHopRetriever instance or None if not available
        """
        try:
            from retrieval.multi_hop_retriever import MultiHopRetriever
            base_retriever = self.container.get_service(IRetriever)
            return MultiHopRetriever(
                base_retriever=base_retriever,
                max_hops=3,
                min_confidence=0.6,
                enable_query_rewriting=True
            )
        except ImportError:
            logger.warning("MultiHopRetriever not available, returning None")
            return None
    
    def create_conflict_detector(self) -> ConflictDetector:
        """Create conflict detector instance.
        
        Returns:
            ConflictDetector instance
        """
        return ConflictDetector()
    
    def create_fallback_guard(self) -> FallbackGuard:
        """Create fallback guard instance.
        
        Returns:
            FallbackGuard instance
        """
        return FallbackGuard()
    
    def create_rag_pipeline(self) -> RAGPipeline:
        """Create complete RAG pipeline with all dependencies.
        
        Returns:
            RAGPipeline instance with full dependency injection
        """
        parser = self.container.get_service(IParser)
        chunker = self.container.get_service(IChunker)
        embedder = self.container.get_service(IEmbedder)
        vector_store = self.container.get_service(IVectorStore)
        retriever = self.container.get_service(IRetriever)
        reranker = self.container.get_service(IReranker)
        refiner = self.container.get_service(IRefiner)
        generator = self.container.get_service(IGenerator)
        safety_checker = self.container.get_service(ConflictDetector)
        
        # Optional advanced services
        query_planner = self.container.get_service(QueryPlanner)
        adaptive_strategy = self.container.get_service(AdaptiveRetrievalStrategy)
        memory_system = self.container.get_service(AIMemorySystem)
        
        return RAGPipeline(
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
            refiner=refiner,
            generator=generator,
            safety_checker=safety_checker,
            query_planner=query_planner,
            adaptive_strategy=adaptive_strategy,
            memory_system=memory_system
        )


def register_services(container: ServiceContainer) -> None:
    """Register all services with the container following SOLID principles.
    
    Args:
        container: Service container instance
    """
    factory = ServiceFactory(container)
    
    # Register singleton services (stateless, thread-safe)
    container.register_singleton(IEmbedder, factory.create_embedder())
    container.register_singleton(IReranker, factory.create_reranker())
    container.register_singleton(IRefiner, factory.create_refiner())
    container.register_singleton(IGenerator, factory.create_generator())
    
    # Register factory services (stateful or configurable)
    container.register_factory(IParser, factory.create_parser)
    container.register_factory(IChunker, factory.create_chunker)
    container.register_factory(IVectorStore, factory.create_vector_store)
    container.register_factory(IRetriever, factory.create_retriever)
    
    # Register engineering services (singletons)
    container.register_singleton(AdaptiveRetrievalStrategy, factory.create_adaptive_strategy())
    container.register_singleton(PipelineTracer, factory.create_pipeline_tracer())
    container.register_singleton(AIMemorySystem, factory.create_memory_system())
    container.register_singleton(RetrievalAnalytics, factory.create_retrieval_analytics())
    container.register_singleton(SystemReliabilityLayer, factory.create_reliability_layer())

    container.register_singleton(ConflictDetector, factory.create_conflict_detector())
    container.register_singleton(FallbackGuard, factory.create_fallback_guard())
    
    # Register engineering services (factories with dependencies)
    container.register_factory(QueryPlanner, factory.create_query_planner)
    container.register_factory(DynamicRetriever, factory.create_dynamic_retriever)
    # Register MultiHopRetriever only if available
    if MultiHopRetriever is not None:
        container.register_factory(MultiHopRetriever, factory.create_multi_hop_retriever)
    container.register_factory(RAGPipeline, factory.create_rag_pipeline)
    
    logger.info("All services registered with container following SOLID principles")
