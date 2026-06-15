"""Core RAG pipeline orchestrator implementing 6-stage architecture with intelligent routing and hallucination prevention."""

import time
import re
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
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
    QueryPlanner = None
    QueryType = None
    AdaptiveRetrievalStrategy = None
    RetrievalMode = None
    logging.warning("Query planning modules not available")

# Import multi-hop retriever
try:
    from retrieval.multi_hop_retriever import MultiHopRetriever
    MULTI_HOP_AVAILABLE = True
except ImportError:
    MULTI_HOP_AVAILABLE = False
    MultiHopRetriever = None
    logging.warning("Multi-hop retriever not available")

# Import memory system
try:
    from core.memory.memory_system import AIMemorySystem
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    AIMemorySystem = None
    logging.warning("Memory system not available")

# Import engineering control services
from core.reliability.system_reliability import SystemReliabilityLayer, ReliabilityCheck
from core.observability.pipeline_tracer import PipelineTracer, PipelineStage
from core.analytics.retrieval_analytics import RetrievalAnalytics
from safety.overshadow_limiter import OvershadowLimiter
from safety.fallback_guard import FallbackGuard

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class ToolRouter:
    """Intelligent router that automatically dispatches queries to Calculator, Database, Web Search, or Opinion Guard using LLM classification."""
    
    def __init__(self):
        """Initialize the Tool Router and register tools."""
        self.tools = {}
        self.client = None
        try:
            from groq import Groq
            from config.settings import settings
            self.api_key = settings.groq_api_key
            self.model = settings.generation.model
            if self.api_key:
                self.client = Groq(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Error importing or initializing Groq in ToolRouter: {e}")
        logger.info("Intelligent LLM Tool Router initialized")
        
    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool instance."""
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")
        
    def classify_and_dispatch(self, query: str, raw_transcript: str = None) -> Optional[str]:
        """Classify query intent and route to the correct tool automatically.
        
        Args:
            query: User query string (may be normalized from multilingual input)
            raw_transcript: Optional raw transcript from voice input for debugging
            
        Returns:
            Factual tool result string, or None if query requires RAG pipeline
        """
        query_lower = query.lower()
        
        # Log routing information for debugging
        if raw_transcript:
            logger.info(f"Raw Transcript: {raw_transcript}")
        logger.info(f"Normalized Query: {query}")
        
        # 1. First attempt: LLM-based agentic classification with multilingual support
        if self.client:
            try:
                import json
                prompt = f"""You are an intelligent query router for a college placement assistant.
Analyze the user's query and decide which tool is best suited to answer it.

Available Tools:
1. "database": For structured query lookups about student records, eligibility checks, list of students placed, roll numbers, GPA cutoffs, or packages.
   Examples: "Who got placed at Google?", "Which student has the highest GPA?", "List companies with package above 10 LPA", "Check eligibility for 22CS010".
2. "web_search": For questions requiring live web lookup, current news, company CEOs, general knowledge, or topics outside our static placement dataset.
   Examples: "Who is the CEO of Google?", "What are the latest hiring trends in 2026?", "Who founded Wipro?", "Wipro CEO evaru?", "Infosys CEO entha?", "What is today's date?", "What time is it?".
3. "calculator": For mathematical operations, conversions (e.g. CGPA to percentage, average calculations).
   Examples: "What is 8.5 CGPA in percentage?", "Calculate average of 5, 8, 12", "Convert 85% to CGPA", "Convert 7.5 CGPA to percentage".
4. "opinion_guard": For career guidance, subjective recommendations, or comparisons between companies.
   Examples: "Should I join TCS or Infosys?", "Compare Google and Amazon", "Which company offers a better career growth?".
5. "rag": For general placement dataset queries, company interview experiences, recruitment distributions, official process details, and general corpus lookup.
   Examples: "What is Google's interview process?", "What rounds does TCS have?", "What is SVECW's placement history?", "TCS CGPA entha?", "Google package entha?".

IMPORTANT: Out-of-scope questions (date, time, weather, general knowledge) should route to WEB_SEARCH, not RAG.

Multilingual Examples (Telugu-English mixed):
- "Wipro CEO evaru" → WEB_SEARCH (Who is the CEO of Wipro?)
- "Infosys package entha" → RAG (What package does Infosys offer?)
- "TCS CGPA entha" → RAG (What is the CGPA requirement for TCS?)
- "Google lo internship unda" → RAG (Does Google offer internships?)
- "Student with CGPA 7 and 1 backlog eligible companies" → DATABASE/RAG
- "Convert 7.5 CGPA to percentage" → CALCULATOR
- "Today date entha" → WEB_SEARCH (What is today's date?)

Choose exactly one tool from: ["database", "web_search", "calculator", "opinion_guard", "rag"].
Respond in JSON format with two keys:
- "tool": The chosen tool name (or "rag" if none of the specific tools are suitable).
- "reason": A brief reason for this decision.
- "confidence": A confidence score between 0.0 and 1.0 for this routing decision.

Query: "{query}"
JSON classification:"""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise query classifier that outputs JSON containing 'tool', 'reason', and 'confidence'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                res_content = response.choices[0].message.content.strip()
                res_data = json.loads(res_content)
                chosen_tool = res_data.get("tool", "rag")
                reason = res_data.get("reason", "")
                confidence = res_data.get("confidence", 0.5)
                
                logger.info(f"LLM Routing Decision: Selected '{chosen_tool}' (Reason: {reason}, Confidence: {confidence})")
                logger.info(f"Selected Tool: {chosen_tool}")
                logger.info(f"Routing Confidence: {confidence}")
                
                if chosen_tool in self.tools:
                    logger.info(f"Routing query to registered tool '{chosen_tool}'")
                    return self.tools[chosen_tool].execute(query)
                elif chosen_tool == "rag":
                    return None
                    
            except Exception as e:
                logger.error(f"LLM Tool Router failed: {e}. Falling back to heuristics.")

        # 2. Heuristic/Regex fallback if LLM routing fails or is unavailable
        # Opinion Guard fallback
        opinion_indicators = [
            "which is better", "should i join", "which is the best", "choose between", 
            "compare", "versus", "vs", "which has better career"
        ]
        if any(indicator in query_lower for indicator in opinion_indicators):
            if "opinion_guard" in self.tools:
                logger.info("Routing query to Opinion Guard (Heuristic)")
                logger.info(f"Selected Tool: opinion_guard")
                logger.info(f"Routing Confidence: 0.8 (heuristic)")
                return self.tools["opinion_guard"].execute(query)
                
        # Calculator fallback
        math_indicators = [
            "average", "mean", "cgpa to percentage", "percentage to cgpa", "convert",
            "calculate", "sum", "divided by", "multiplied by", "subtract", "plus"
        ]
        has_cgpa = "cgpa" in query_lower
        has_percent = "percent" in query_lower or "percentage" in query_lower or "%" in query_lower
        is_cgpa_percent = has_cgpa and has_percent
        if (any(ind in query_lower for ind in math_indicators) or is_cgpa_percent) and any(c.isdigit() for c in query_lower):
            if "calculator" in self.tools:
                logger.info("Routing query to Calculator Tool (Heuristic)")
                logger.info(f"Selected Tool: calculator")
                logger.info(f"Routing Confidence: 0.8 (heuristic)")
                return self.tools["calculator"].execute(query)
                
        # Database fallback
        db_indicators = [
            "list companies", "which companies allow", "companies with cgpa", "cutoff above",
            "cutoff below", "package above", "package below", "allow at least", "no backlog", "zero backlog",
            "placed", "student", "roll number"
        ]
        if any(ind in query_lower for ind in db_indicators):
            if "database" in self.tools:
                logger.info("Routing query to Database Tool (Heuristic)")
                logger.info(f"Selected Tool: database")
                logger.info(f"Routing Confidence: 0.8 (heuristic)")
                return self.tools["database"].execute(query)
        
        # Web Search fallback
        web_search_indicators = [
            "ceo", "founder", "latest", "current", "news", "trends", "questions asked",
            "interview questions", "dsa questions", "top questions", "common questions",
            "hiring trends", "salary trends", "market trends", "outside", "external"
        ]
        
        # Out-of-scope/general knowledge indicators
        out_of_scope_indicators = [
            "today", "date", "time", "weather", "temperature", "what day", "what month",
            "what year", "current date", "current time", "now", "capital", "population",
            "who is president", "who is prime minister", "country", "city"
        ]
        
        if any(indicator in query_lower for indicator in out_of_scope_indicators):
            if "web_search" in self.tools:
                logger.info("Routing query to Web Search Tool (Out-of-scope question)")
                logger.info(f"Selected Tool: web_search")
                logger.info(f"Routing Confidence: 0.9 (heuristic)")
                return self.tools["web_search"].execute(query)
        
        if any(indicator in query_lower for indicator in web_search_indicators):
            if "web_search" in self.tools:
                logger.info("Routing query to Web Search Tool (Heuristic)")
                logger.info(f"Selected Tool: web_search")
                logger.info(f"Routing Confidence: 0.8 (heuristic)")
                return self.tools["web_search"].execute(query)
        
        logger.info("No specific tool matched, routing to RAG pipeline")
        logger.info(f"Selected Tool: RAG")
        logger.info(f"Routing Confidence: 0.5 (default)")
        return None


class RAGPipeline:
    """6-stage RAG orchestrator for placement intelligence with intelligent routing, tracing, and hallucination guards."""
    
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
        query_planner: Optional['QueryPlanner'] = None,
        adaptive_strategy: Optional['AdaptiveRetrievalStrategy'] = None,
        memory_system: Optional['AIMemorySystem'] = None,
        multi_hop_retriever: Optional['MultiHopRetriever'] = None,
        pipeline_tracer: Optional[PipelineTracer] = None,
        retrieval_analytics: Optional[RetrievalAnalytics] = None,
        reliability_layer: Optional[SystemReliabilityLayer] = None,
        overshadow_limiter: Optional[OvershadowLimiter] = None,
        fallback_guard: Optional[FallbackGuard] = None
    ):
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
        
        # Instantiate and wire Observability, Reliability & Truncation Layer
        self.pipeline_tracer = pipeline_tracer or PipelineTracer()
        self.retrieval_analytics = retrieval_analytics or RetrievalAnalytics()
        self.reliability_layer = reliability_layer or SystemReliabilityLayer()
        self.overshadow_limiter = overshadow_limiter or OvershadowLimiter()
        self.fallback_guard = fallback_guard or FallbackGuard()
        
        # Initialize multi-hop retriever if not provided but available
        if self.multi_hop_retriever is None and MULTI_HOP_AVAILABLE and MultiHopRetriever is not None:
            self.multi_hop_retriever = MultiHopRetriever(
                base_retriever=retriever,
                max_hops=2,
                min_confidence=0.6,
                enable_query_rewriting=True
            )
            
        # Initialize Tool Router and register tools
        self.tool_router = ToolRouter()
        
        from core.tools.calculator import CalculatorTool
        from core.tools.database_tool import DatabaseTool
        from core.tools.opinion_guard import OpinionGuard
        from core.tools.web_search import WebSearchTool
        
        self.tool_router.register_tool("calculator", CalculatorTool())
        self.tool_router.register_tool("database", DatabaseTool())
        self.tool_router.register_tool("opinion_guard", OpinionGuard())
        self.tool_router.register_tool("web_search", WebSearchTool())
        
        logger.info("RAG Pipeline initialized with LLM Tool Router and WebSearch.")
    
    def ingest(self, file_path: str) -> None:
        """Stage 0: Parse → Chunk → Embed → Index."""
        logger.info(f"Starting ingestion for {file_path}")
        
        documents = self.parser.parse(file_path)
        logger.info(f"Parsed {len(documents)} documents")
        
        chunks = self.chunker.chunk(documents)
        logger.info(f"Generated {len(chunks)} chunks")
        
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedder.embed_documents(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
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
        """Execute full RAG pipeline with intelligent routing, memory, and hallucination checks."""
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        # Start Trace
        trace_id = self.pipeline_tracer.start_trace(query)
        trace_stages = []
        
        # ── STAGE 0: Intelligent Tool Routing ─────────────────────────────────
        self.pipeline_tracer.start_stage(PipelineStage.QUERY_PLANNING, {"query": query})
        
        # Check if this is a voice query with raw transcript
        raw_transcript = None
        routing_confidence = 0.5  # Default routing confidence
        
        # Check session state for voice transcript (for Streamlit UI)
        try:
            import streamlit as st
            if 'voice_transcript' in st.session_state and st.session_state.voice_transcript:
                raw_transcript = st.session_state.voice_transcript
        except:
            pass
        
        tool_result = self.tool_router.classify_and_dispatch(query, raw_transcript)
        
        # Extract routing confidence from logs (simplified approach)
        # In a production system, we'd return this from classify_and_dispatch
        routing_confidence = 0.8 if tool_result else 0.5
        
        self.pipeline_tracer.end_stage(PipelineStage.QUERY_PLANNING, {"routed_to_tool": tool_result is not None})
        trace_stages.append("query_planning")
        
        if tool_result:
            latency_ms = (time.time() - start_time) * 1000
            
            # Trace Tool Execution
            self.pipeline_tracer.start_stage(PipelineStage.GENERATION, {"tool": "execution"})
            self.pipeline_tracer.end_stage(PipelineStage.GENERATION, {"answer": tool_result})
            trace_stages.append("generation")
            
            self.pipeline_tracer.set_query_info("tool_query", "direct_lookup")
            self.pipeline_tracer.set_final_answer(tool_result, 1.0)
            self.pipeline_tracer.end_trace(success=True)
            
            # Log Tool execution in analytics
            self.retrieval_analytics.track_retrieval(
                query=query,
                retrieved_docs=[],
                reranked_docs=[],
                confidence_score=1.0,
                reranking_scores=[],
                token_usage=0,
                metadata={"tool_execution": True, "latency_ms": latency_ms}
            )
            
            return {
                "answer": tool_result,
                "sources": [],
                "confidence": 1.0,
                "conflicts": 0,
                "query_type": "tool_query",
                "retrieval_mode": "direct_lookup",
                "cached": False,
                "reliability": {
                    "passed": True,
                    "verdict": "PASS",
                    "confidence": 1.0,
                    "issues": [],
                    "groundedness_score": 1.0,
                    "consistency_score": 1.0,
                    "lookback_ratio": 1.0
                }
            }
            
        # ── STAGE 0.5: Semantic Cache Check ────────────────────────────────────
        if self.memory_system and MEMORY_AVAILABLE:
            cached_result = self.memory_system.semantic_cache.get(query)
            if cached_result:
                logger.info("Query found in semantic cache, returning cached result")
                latency_ms = (time.time() - start_time) * 1000
                
                self.pipeline_tracer.set_query_info("cached_query", "cached")
                self.pipeline_tracer.set_final_answer(cached_result["answer"], 0.9)
                self.pipeline_tracer.end_trace(success=True)
                
                return {
                    "answer": cached_result["answer"],
                    "sources": cached_result.get("sources", []),
                    "confidence": cached_result.get("confidence", 0.9),
                    "cached": True,
                    "query_type": "cached",
                    "retrieval_mode": "cached",
                    "reliability": {
                        "passed": True,
                        "verdict": "PASS",
                        "confidence": 0.9,
                        "issues": [],
                        "groundedness_score": 1.0,
                        "consistency_score": 1.0,
                        "lookback_ratio": 1.0
                    }
                }
        
        # ── STAGE 1: Adaptive Query Intent & Planning ──────────────────────────
        query_type = None
        retrieval_mode = None
        if self.query_planner and QUERY_PLANNING_AVAILABLE:
            query_plan = self.query_planner.plan(query)
            query_type = query_plan.query_type
            retrieval_mode = query_plan.retrieval_strategy
            logger.info(f"Query classified as: {query_type.value}, retrieval mode: {retrieval_mode}")
        
        # ── STAGE 2: Retrieval (Adaptive vs Multi-Hop) ─────────────────────────
        self.pipeline_tracer.start_stage(PipelineStage.RETRIEVAL)
        
        if retrieval_mode and self.adaptive_strategy:
            retrieved = self._adaptive_retrieve(query, retrieval_mode, query_type)
        elif QUERY_PLANNING_AVAILABLE and query_type == QueryType.MULTI_HOP and self.multi_hop_retriever:
            logger.info("Using multi-hop retrieval for query")
            retrieved = self.multi_hop_retriever.retrieve(query, k=settings.retrieval.top_k_dense + 5)
            trace_stages.append("query_rewriting")
        else:
            retrieved = self.retriever.retrieve(query, k=settings.retrieval.top_k_dense)
            
        self.pipeline_tracer.end_stage(PipelineStage.RETRIEVAL, {"retrieved_count": len(retrieved)})
        trace_stages.append("retrieval")
        
        # Metadata Filtering
        if query_type and QUERY_PLANNING_AVAILABLE:
            retrieved = self._filter_by_metadata(retrieved, query_type)
            
        # ── STAGE 3: Reranking ────────────────────────────────────────────────
        self.pipeline_tracer.start_stage(PipelineStage.RERANKING)
        reranked = self.reranker.rerank(query, retrieved, top_k=settings.retrieval.top_k_final)
        self.pipeline_tracer.end_stage(PipelineStage.RERANKING, {"reranked_count": len(reranked)})
        trace_stages.append("reranking")
        
        # ── STAGE 4: Context Refinement ───────────────────────────────────────
        self.pipeline_tracer.start_stage(PipelineStage.REFINEMENT)
        refined = self.refiner.refine(reranked)
        self.pipeline_tracer.end_stage(PipelineStage.REFINEMENT, {"refined_count": len(refined)})
        trace_stages.append("refinement")
        
        # ── STAGE 4.5: System 2 Attention Context Filtering ────────────────────
        # Disabled for performance - comment out to enable
        # self.pipeline_tracer.start_stage(PipelineStage.QUERY_REWRITING, {"before_s2a": len(refined)})
        # refined = self.reliability_layer.apply_s2a(query, refined)
        # self.pipeline_tracer.end_stage(PipelineStage.QUERY_REWRITING, {"after_s2a": len(refined)})
        # trace_stages.append("query_rewriting")
        
        # ── STAGE 4.6: Overshadow Limiter & Token Budgeting ──────────────────
        refined, overshadow_risk = self.overshadow_limiter.limit_context(refined)
        
        # ── STAGE 5: Safety Checks ────────────────────────────────────────────
        self.pipeline_tracer.start_stage(PipelineStage.VALIDATION)
        conflicts = self.safety_checker.check_conflict(refined) if self.safety_checker else []
        if conflicts and settings.safety.enable_conflict_detection:
            logger.warning(f"Found {len(conflicts)} conflicting documents")
            
        out_of_corpus = self.safety_checker.check_out_of_corpus(query, refined) if self.safety_checker else False
        self.pipeline_tracer.end_stage(PipelineStage.VALIDATION, {"out_of_corpus": out_of_corpus})
        trace_stages.append("validation")
        
        if out_of_corpus and settings.safety.enable_fallback_guard:
            logger.warning("Query may be out of corpus scope")
            answer = "I don't have enough information in the placement documents to answer this question accurately."
            self.pipeline_tracer.end_trace(success=False, error_message="Out of corpus")
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.0,
                "out_of_corpus": True,
                "query_type": query_type.value if query_type else "unknown"
            }
            
        # ── STAGE 6: Generation (Self-Consistency) ────────────────────────────
        self.pipeline_tracer.start_stage(PipelineStage.GENERATION)
        
        # We execute the self-consistency loop to sample multiple answers and select the best
        consistency_res = self.reliability_layer.consistency_verifier.verify(query, refined, self.generator)
        answer = consistency_res["best_answer"]
        consistency_score = consistency_res.get("consistency_score", 1.0)
        
        self.pipeline_tracer.end_stage(PipelineStage.GENERATION, {"answer": answer})
        trace_stages.append("generation")
        
        # ── STAGE 7: Factual Recitation Checking & Reliability Verdict ─────────
        self.pipeline_tracer.start_stage(PipelineStage.RELIABILITY_CHECK)
        
        base_confidence = self._calculate_confidence(refined, query, routing_confidence)
        reliability_report = self.reliability_layer.check_reliability(
            query=query,
            answer=answer,
            context=refined,
            confidence=base_confidence,
            query_type=query_type.value if hasattr(query_type, 'value') else str(query_type) if query_type else "factual",
            trace_stages=trace_stages,
            generator=self.generator
        )
        
        self.pipeline_tracer.end_stage(PipelineStage.RELIABILITY_CHECK, {
            "verdict": reliability_report.verdict,
            "groundedness": reliability_report.groundedness_score
        })
        trace_stages.append("reliability_check")
        
        # Handle Failures
        if reliability_report.fallback_triggered:
            logger.warning(f"Reliability failure. Triggering fallback. Reason: {reliability_report.fallback_reason}")
            answer = self.reliability_layer.get_fallback_response(reliability_report.fallback_reason)
            
        # Validate and fix inline citations post-generation
        answer = self._validate_and_fix_citations(answer, refined)
        
        # End Trace
        self.pipeline_tracer.set_query_info(
            query_type.value if hasattr(query_type, 'value') else str(query_type) if query_type else "unknown",
            retrieval_mode.value if hasattr(retrieval_mode, 'value') else str(retrieval_mode) if retrieval_mode else "standard"
        )
        self.pipeline_tracer.set_final_answer(answer, reliability_report.confidence)
        self.pipeline_tracer.end_trace(success=reliability_report.passed)
        
        # Log analytics
        latency_ms = (time.time() - start_time) * 1000
        rerank_scores = [doc.metadata.get("rerank_score", 0.5) for doc in refined]
        self.retrieval_analytics.track_retrieval(
            query=query,
            retrieved_docs=retrieved,
            reranked_docs=refined,
            confidence_score=reliability_report.confidence,
            reranking_scores=rerank_scores,
            token_usage=sum(len(doc.page_content) // 4 for doc in refined),
            metadata={
                "overshadow_risk": overshadow_risk,
                "latency_ms": latency_ms,
                "verdict": reliability_report.verdict,
                "groundedness": reliability_report.groundedness_score,
                "consistency": reliability_report.consistency_score
            }
        )
        
        # Update Memory Systems
        if self.memory_system and MEMORY_AVAILABLE:
            self.memory_system.semantic_cache.set(query, answer, {
                "confidence": reliability_report.confidence,
                "sources": [{"text": doc.page_content, "metadata": doc.metadata} for doc in refined]
            })
            self.memory_system.conversation_memory.add_turn(query, answer)
            
        return {
            "answer": answer,
            "sources": [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in refined
            ],
            "confidence": reliability_report.confidence,
            "conflicts": len(conflicts) if conflicts else 0,
            "query_type": query_type.value if hasattr(query_type, 'value') else str(query_type) if query_type else "unknown",
            "retrieval_mode": retrieval_mode.value if hasattr(retrieval_mode, 'value') else str(retrieval_mode) if retrieval_mode else "standard",
            "cached": False,
            "reliability": {
                "passed": reliability_report.passed,
                "verdict": reliability_report.verdict,
                "confidence": reliability_report.confidence,
                "issues": reliability_report.issues,
                "groundedness_score": reliability_report.groundedness_score,
                "consistency_score": reliability_report.consistency_score,
                "overshadow_risk": overshadow_risk,
                "lookback_ratio": reliability_report.lookback_ratio
            }
        }
        
    def _validate_and_fix_citations(self, answer: str, context: List[Document]) -> str:
        """Validate all [Source X] citations in the answer against context bounds, correcting mismatches."""
        if not context:
            return re.sub(r'\[Source \d+\]', '', answer)
            
        citations = re.findall(r'\[Source (\d+)\]', answer)
        fixed_answer = answer
        
        for cit_str in set(citations):
            idx = int(cit_str)
            if idx < 1 or idx > len(context):
                # We have an invalid source reference! Match sentence to find closest context document
                sentences = re.split(r'[.!?]', fixed_answer)
                for sentence in sentences:
                    if f"[Source {cit_str}]" in sentence:
                        best_match_idx = 0
                        best_overlap = -1
                        sentence_words = set(sentence.lower().split())
                        for doc_idx, doc in enumerate(context):
                            doc_words = set(doc.page_content.lower().split())
                            overlap = len(sentence_words & doc_words)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_match_idx = doc_idx
                        # Correct citation
                        fixed_answer = fixed_answer.replace(f"[Source {cit_str}]", f"[Source {best_match_idx + 1}]")
                        break
        return fixed_answer
    
    def _adaptive_retrieve(self, query: str, retrieval_mode, query_type) -> List[Document]:
        """Perform adaptive retrieval."""
        mode_str = retrieval_mode.value if hasattr(retrieval_mode, 'value') else str(retrieval_mode)
        
        if mode_str == "semantic_heavy":
            k = settings.retrieval.top_k_dense + 5
            retrieved = self.retriever.retrieve(query, k=k)
        elif mode_str == "keyword_heavy":
            k = settings.retrieval.top_k_dense
            retrieved = self.retriever.retrieve(query, k=k)
        elif mode_str == "metadata_first":
            k = settings.retrieval.top_k_dense + 10
            retrieved = self.retriever.retrieve(query, k=k)
        elif QUERY_PLANNING_AVAILABLE and mode_str == "multi_hop" and self.multi_hop_retriever:
            logger.info("Using multi-hop retrieval for multi_hop retrieval mode")
            retrieved = self.multi_hop_retriever.retrieve(query, k=settings.retrieval.top_k_dense + 5)
        else:
            retrieved = self.retriever.retrieve(query, k=settings.retrieval.top_k_dense)
        
        return retrieved
    
    def _filter_by_metadata(self, documents: List[Document], query_type) -> List[Document]:
        """Filter documents by metadata based on query type."""
        filtered = []
        
        if not QUERY_PLANNING_AVAILABLE or query_type is None:
            return documents
        
        for doc in documents:
            metadata = doc.metadata
            
            if query_type == QueryType.ELIGIBILITY:
                if metadata.get("type") in ["eligibility", "requirements", "criteria", "qualification"]:
                    filtered.append(doc)
            elif query_type == QueryType.INTERNSHIP:
                if metadata.get("type") in ["internship", "placement", "offers", "stipend"]:
                    filtered.append(doc)
            elif query_type == QueryType.STATISTICS:
                if metadata.get("type") in ["statistics", "package", "salary", "data", "compensation"]:
                    filtered.append(doc)
            elif query_type == QueryType.INTERVIEW:
                if metadata.get("type") in ["interview", "experience", "process", "round", "technical", "hr"]:
                    filtered.append(doc)
            elif query_type == QueryType.PLACEMENT_TREND:
                if metadata.get("type") in ["statistics", "trend", "data", "year", "annual"]:
                    filtered.append(doc)
            else:
                filtered.append(doc)
        
        if not filtered:
            return documents
            
        return filtered
    
    def _calculate_confidence(self, documents: List[Document], query: str = None, routing_confidence: float = 0.5) -> float:
        """Calculate confidence score based on retrieved documents and routing confidence.
        
        Separates retrieval confidence from answer confidence:
        - Retrieval confidence: Based on document count and relevance scores
        - Answer confidence: Based on routing confidence and semantic match
        
        Args:
            documents: Retrieved documents
            query: Original query for semantic relevance check
            routing_confidence: Confidence score from the routing layer
            
        Returns:
            Combined confidence score
        """
        if not documents:
            return 0.0
        
        # Base retrieval confidence based on document count
        retrieval_confidence = min(len(documents) / settings.retrieval.top_k_final, 1.0)
        
        # Consider rerank scores if available
        if documents and hasattr(documents[0], 'metadata'):
            rerank_scores = [doc.metadata.get("rerank_score", 0.5) for doc in documents]
            avg_rerank_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.5
            # Weight retrieval confidence by average rerank score
            retrieval_confidence = retrieval_confidence * avg_rerank_score
        
        # Combine retrieval confidence with routing confidence
        # If routing confidence is low (e.g., uncertain routing to RAG), reduce overall confidence
        combined_confidence = (retrieval_confidence * 0.7) + (routing_confidence * 0.3)
        
        # Log confidence breakdown for debugging
        logger.info(f"Confidence Breakdown - Retrieval: {retrieval_confidence:.2f}, Routing: {routing_confidence:.2f}, Combined: {combined_confidence:.2f}")
        
        return combined_confidence
