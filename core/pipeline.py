"""Core RAG pipeline orchestrator implementing 6-stage architecture with intelligent routing."""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from core.interfaces import (
    IParser, IChunker, IEmbedder, IVectorStore,
    IRetriever, IReranker, IRefiner, IGenerator,
    ISafetyChecker
)
from config.settings import settings
import logging

# Import query planning and adaptive retrieval
try:
    from core.query_planning.query_planner import QueryPlanner, QueryType
    from core.retrieval.adaptive_strategy import AdaptiveRetrievalStrategy, RetrievalMode
    QUERY_PLANNING_AVAILABLE = True
except ImportError:
    QUERY_PLANNING_AVAILABLE = False
    logging.warning("Query planning modules not available")

# Import multi-hop retriever
try:
    from retrieval.multi_hop_retriever import MultiHopRetriever
    MULTI_HOP_AVAILABLE = True
except ImportError:
    MULTI_HOP_AVAILABLE = False
    logging.warning("Multi-hop retriever not available")

# Import memory system
try:
    from core.memory.memory_system import AIMemorySystem
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logging.warning("Memory system not available")

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """6-stage RAG orchestrator for placement intelligence with intelligent routing."""
    
    def __init__(
        self,
        parser: IParser,
        chunker: IChunker,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        retriever: IRetriever,
        reranker: IReranker,
        refiner: IRefiner,
        generator: IGenerator,
        safety_checker: ISafetyChecker,
        query_planner: Optional[QueryPlanner] = None,
        adaptive_strategy: Optional[AdaptiveRetrievalStrategy] = None,
        memory_system: Optional[AIMemorySystem] = None,
        multi_hop_retriever: Optional[MultiHopRetriever] = None
    ):
        """Initialize the RAG pipeline with all components.
        
        Args:
            parser: Document parser
            chunker: Document chunker
            embedder: Embedding generator
            vector_store: Vector storage
            retriever: Document retriever
            reranker: Document reranker
            refiner: Context refiner
            generator: Answer generator
            safety_checker: Safety validation
            query_planner: Optional query planner for intelligent routing
            adaptive_strategy: Optional adaptive retrieval strategy
            memory_system: Optional memory system for caching and conversation history
            multi_hop_retriever: Optional multi-hop retriever for complex queries
        """
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker
        self.refiner = refiner
        self.generator = generator
        self.safety_checker = safety_checker
        self.query_planner = query_planner
        self.adaptive_strategy = adaptive_strategy
        self.memory_system = memory_system
        self.multi_hop_retriever = multi_hop_retriever
        
        # Initialize multi-hop retriever if not provided but available
        if self.multi_hop_retriever is None and MULTI_HOP_AVAILABLE:
            self.multi_hop_retriever = MultiHopRetriever(
                base_retriever=retriever,
                max_hops=3,
                min_confidence=0.6,
                enable_query_rewriting=True
            )
        
        logger.info("RAG Pipeline initialized with all components including multi-hop support")
    
    def ingest(self, file_path: str) -> None:
        """Stage 0: Parse → Chunk → Embed → Index.
        
        Args:
            file_path: Path to document file
        """
        logger.info(f"Starting ingestion for {file_path}")
        
        # Stage 0.1: Parse
        documents = self.parser.parse(file_path)
        logger.info(f"Parsed {len(documents)} documents")
        
        # Stage 0.2: Chunk
        chunks = self.chunker.chunk(documents)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Stage 0.3: Embed
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedder.embed_documents(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Stage 0.4: Index
        dict_chunks = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in chunks
        ]
        self.vector_store.add_documents(dict_chunks)
        logger.info("Documents indexed successfully")
    
    def query(self, query: str) -> Dict[str, Any]:
        """Execute full RAG pipeline with intelligent routing and memory: Memory Check → Query Planning → Retrieval → Reranking → Refinement → Generation → Safety → Memory Update.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Stage 0.5: Check semantic cache (if available)
        if self.memory_system and MEMORY_AVAILABLE:
            cached_result = self.memory_system.semantic_cache.get(query)
            if cached_result:
                logger.info("Query found in semantic cache, returning cached result")
                return {
                    "answer": cached_result["answer"],
                    "sources": cached_result.get("sources", []),
                    "confidence": cached_result.get("confidence", 0.9),
                    "cached": True,
                    "query_type": "cached",
                    "retrieval_mode": "cached"
                }
        
        # Stage 0: Query Intent Classification (if available)
        query_type = None
        retrieval_mode = None
        if self.query_planner and QUERY_PLANNING_AVAILABLE:
            query_plan = self.query_planner.plan(query)
            query_type = query_plan.query_type
            retrieval_mode = query_plan.retrieval_strategy
            logger.info(f"Query classified as: {query_type.value}, retrieval mode: {retrieval_mode}")
        
        # Stage 1: Retrieval with adaptive mode selection
        if retrieval_mode and self.adaptive_strategy:
            # Use adaptive retrieval based on query type
            retrieved = self._adaptive_retrieve(query, retrieval_mode, query_type)
        elif query_type == QueryType.MULTI_HOP and self.multi_hop_retriever:
            # Use multi-hop retrieval for complex queries
            logger.info("Using multi-hop retrieval for MULTI_HOP query type")
            retrieved = self.multi_hop_retriever.retrieve(query, k=settings.retrieval.top_k_dense + 5)
        else:
            # Standard retrieval
            retrieved = self.retriever.retrieve(query, k=settings.retrieval.top_k_dense)
        logger.info(f"Retrieved {len(retrieved)} documents")
        
        # Stage 2: Metadata filtering based on query type
        if query_type and QUERY_PLANNING_AVAILABLE:
            retrieved = self._filter_by_metadata(retrieved, query_type)
            logger.info(f"After metadata filtering: {len(retrieved)} documents")
        
        # Stage 3: Reranking
        reranked = self.reranker.rerank(query, retrieved, top_k=settings.retrieval.top_k_final)
        logger.info(f"Reranked to {len(reranked)} documents")
        
        # Stage 4: Context Refinement
        refined = self.refiner.refine(reranked)
        logger.info(f"Refined to {len(refined)} documents")
        
        # Stage 5: Safety Check
        conflicts = self.safety_checker.check_conflict(refined)
        if conflicts and settings.safety.enable_conflict_detection:
            logger.warning(f"Found {len(conflicts)} conflicting documents")
        
        out_of_corpus = self.safety_checker.check_out_of_corpus(query, refined)
        if out_of_corpus and settings.safety.enable_fallback_guard:
            logger.warning("Query may be out of corpus scope")
            return {
                "answer": "I don't have enough information in the placement documents to answer this question accurately.",
                "sources": [],
                "confidence": 0.0,
                "out_of_corpus": True,
                "query_type": query_type.value if query_type else "unknown"
            }
        
        # Stage 6: Generation
        answer = self.generator.generate(query, refined)
        logger.info("Generated answer")
        
        # Stage 7: Update memory (if available)
        if self.memory_system and MEMORY_AVAILABLE:
            # Store in semantic cache
            self.memory_system.semantic_cache.set(query, {
                "answer": answer,
                "sources": [{"text": doc.page_content, "metadata": doc.metadata} for doc in refined],
                "confidence": self._calculate_confidence(refined)
            })
            
            # Add to conversation history
            self.memory_system.conversation_memory.add_turn(query, answer)
            
            logger.info("Updated memory with query-answer pair")
        
        # Stage 8: Return results
        return {
            "answer": answer,
            "sources": [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in refined
            ],
            "confidence": self._calculate_confidence(refined),
            "conflicts": len(conflicts) if conflicts else 0,
            "query_type": query_type.value if hasattr(query_type, 'value') else str(query_type) if query_type else "unknown",
            "retrieval_mode": retrieval_mode.value if hasattr(retrieval_mode, 'value') else str(retrieval_mode) if retrieval_mode else "standard",
            "cached": False
        }
    
    def _adaptive_retrieve(self, query: str, retrieval_mode, query_type) -> List[Document]:
        """Perform adaptive retrieval based on query type and mode with multi-hop support.
        
        Args:
            query: User query
            retrieval_mode: Retrieval mode to use (string or enum)
            query_type: Query type for context
            
        Returns:
            Retrieved documents
        """
        # Convert to string if it's an enum
        mode_str = retrieval_mode.value if hasattr(retrieval_mode, 'value') else str(retrieval_mode)
        
        # Adjust retrieval parameters based on mode
        if mode_str == "semantic_heavy":
            # Increase semantic retrieval, decrease keyword
            k = settings.retrieval.top_k_dense + 5
            retrieved = self.retriever.retrieve(query, k=k)
        elif mode_str == "keyword_heavy":
            # Increase keyword retrieval
            k = settings.retrieval.top_k_dense
            retrieved = self.retriever.retrieve(query, k=k)
        elif mode_str == "metadata_first":
            # Prioritize metadata filtering
            k = settings.retrieval.top_k_dense + 10
            retrieved = self.retriever.retrieve(query, k=k)
        elif mode_str == "multi_hop" and self.multi_hop_retriever:
            # Use multi-hop retrieval
            logger.info("Using multi-hop retrieval for multi_hop retrieval mode")
            retrieved = self.multi_hop_retriever.retrieve(query, k=settings.retrieval.top_k_dense + 5)
        else:
            # Balanced hybrid
            retrieved = self.retriever.retrieve(query, k=settings.retrieval.top_k_dense)
        
        return retrieved
    
    def _filter_by_metadata(self, documents: List[Document], query_type: QueryType) -> List[Document]:
        """Filter documents by metadata based on query type with enhanced type mapping.
        
        Args:
            documents: Retrieved documents
            query_type: Query type for filtering
            
        Returns:
            Filtered documents
        """
        filtered = []
        
        for doc in documents:
            metadata = doc.metadata
            
            # Filter based on query type with enhanced type mapping
            if query_type == QueryType.ELIGIBILITY:
                # Prioritize eligibility-related chunks
                if metadata.get("type") in ["eligibility", "requirements", "criteria", "qualification"]:
                    filtered.append(doc)
            elif query_type == QueryType.INTERNSHIP:
                # Prioritize internship-related chunks
                if metadata.get("type") in ["internship", "placement", "offers", "stipend"]:
                    filtered.append(doc)
            elif query_type == QueryType.STATISTICS:
                # Prioritize statistics-related chunks
                if metadata.get("type") in ["statistics", "package", "salary", "data", "compensation"]:
                    filtered.append(doc)
            elif query_type == QueryType.INTERVIEW:
                # Prioritize interview-related chunks
                if metadata.get("type") in ["interview", "experience", "process", "round", "technical", "hr"]:
                    filtered.append(doc)
            elif query_type == QueryType.PLACEMENT_TREND:
                # Prioritize trend-related chunks
                if metadata.get("type") in ["statistics", "trend", "data", "year", "annual"]:
                    filtered.append(doc)
            else:
                # No specific filtering for other types
                filtered.append(doc)
        
        # If filtering removed all documents, return original
        if not filtered:
            logger.warning(f"Metadata filtering removed all documents for type {query_type.value}, returning original")
            return documents
        
        logger.info(f"Metadata filtering: {len(documents)} -> {len(filtered)} documents for type {query_type.value}")
        return filtered
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0
        
        # Simple confidence based on number of relevant documents
        return min(len(documents) / settings.retrieval.top_k_final, 1.0)


class ToolRouter:
    """Router for tool-augmented agent capabilities."""
    
    def __init__(self):
        """Initialize tool router with available tools."""
        self.tools = {}
        logger.info("Tool Router initialized")
    
    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool with the router.
        
        Args:
            name: Tool name
            tool: Tool instance
        """
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")
    
    def dispatch(self, query: str) -> Optional[Any]:
        """Dispatch query to appropriate tool.
        
        Args:
            query: User query
            
        Returns:
            Tool result or None if no tool matches
        """
        # Simple keyword-based routing
        for name, tool in self.tools.items():
            if name.lower() in query.lower():
                logger.info(f"Dispatching to tool: {name}")
                return tool.execute(query)
        
        return None
