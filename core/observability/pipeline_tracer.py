"""Pipeline Tracing System for comprehensive retrieval observability."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import time
from enum import Enum
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stage enumeration."""
    QUERY_PLANNING = "query_planning"
    QUERY_REWRITING = "query_rewriting"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    REFINEMENT = "refinement"
    GENERATION = "generation"
    VALIDATION = "validation"
    RELIABILITY_CHECK = "reliability_check"


@dataclass
class StageTrace:
    """Trace information for a single pipeline stage."""
    stage: PipelineStage
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""


@dataclass
class ChunkTrace:
    """Trace information for a retrieved chunk."""
    chunk_id: str
    content: str
    source: str
    page_number: Optional[int] = None
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    selected: bool = True
    discard_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    """Complete trace of a query through the pipeline."""
    trace_id: str
    query: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    stages: List[StageTrace] = field(default_factory=list)
    chunks: List[ChunkTrace] = field(default_factory=list)
    final_answer: str = ""
    confidence_score: float = 0.0
    query_type: str = ""
    retrieval_strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""


class PipelineTracer:
    """Traces execution through the RAG pipeline."""
    
    def __init__(self, trace_dir: str = "data/traces"):
        """Initialize the pipeline tracer.
        
        Args:
            trace_dir: Directory to store trace files
        """
        self.trace_dir = trace_dir
        self.current_trace: Optional[PipelineTrace] = None
        self.trace_history: List[PipelineTrace] = []
        
        os.makedirs(trace_dir, exist_ok=True)
        
        logger.info("PipelineTracer initialized")
    
    def start_trace(self, query: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new pipeline trace.
        
        Args:
            query: User query
            metadata: Optional metadata
            
        Returns:
            Trace ID
        """
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.current_trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        logger.info(f"Started trace: {trace_id}")
        return trace_id
    
    def end_trace(self, success: bool = True, error_message: str = "") -> None:
        """End the current pipeline trace.
        
        Args:
            success: Whether the pipeline execution was successful
            error_message: Error message if failed
        """
        if not self.current_trace:
            logger.warning("No active trace to end")
            return
        
        self.current_trace.end_time = datetime.now()
        self.current_trace.total_duration_ms = (
            self.current_trace.end_time - self.current_trace.start_time
        ).total_seconds() * 1000
        self.current_trace.success = success
        self.current_trace.error_message = error_message
        
        # Save trace
        self._save_trace(self.current_trace)
        
        # Add to history
        self.trace_history.append(self.current_trace)
        
        logger.info(f"Ended trace: {self.current_trace.trace_id}")
        self.current_trace = None
    
    def start_stage(
        self,
        stage: PipelineStage,
        input_data: Dict[str, Any] = None
    ) -> None:
        """Start tracing a pipeline stage.
        
        Args:
            stage: Pipeline stage
            input_data: Input data for the stage
        """
        if not self.current_trace:
            logger.warning("No active trace")
            return
        
        stage_trace = StageTrace(
            stage=stage,
            start_time=datetime.now(),
            input_data=input_data or {}
        )
        
        self.current_trace.stages.append(stage_trace)
        logger.info(f"Started stage: {stage.value}")
    
    def end_stage(
        self,
        stage: PipelineStage,
        output_data: Dict[str, Any] = None,
        success: bool = True,
        error_message: str = ""
    ) -> None:
        """End tracing a pipeline stage.
        
        Args:
            stage: Pipeline stage
            output_data: Output data from the stage
            success: Whether stage execution was successful
            error_message: Error message if failed
        """
        if not self.current_trace:
            logger.warning("No active trace")
            return
        
        # Find the stage trace
        stage_trace = None
        for st in reversed(self.current_trace.stages):
            if st.stage == stage and st.end_time is None:
                stage_trace = st
                break
        
        if not stage_trace:
            logger.warning(f"No active stage found: {stage.value}")
            return
        
        stage_trace.end_time = datetime.now()
        stage_trace.duration_ms = (
            stage_trace.end_time - stage_trace.start_time
        ).total_seconds() * 1000
        stage_trace.output_data = output_data or {}
        stage_trace.success = success
        stage_trace.error_message = error_message
        
        logger.info(f"Ended stage: {stage.value} ({stage_trace.duration_ms:.2f}ms)")
    
    def trace_chunks(
        self,
        retrieved_docs: List[Document],
        reranked_docs: List[Document] = None,
        discarded_docs: List[Document] = None
    ) -> None:
        """Trace chunk retrieval and selection.
        
        Args:
            retrieved_docs: Initially retrieved documents
            reranked_docs: Reranked documents
            discarded_docs: Discarded documents
        """
        if not self.current_trace:
            logger.warning("No active trace")
            return
        
        # Trace retrieved chunks
        for i, doc in enumerate(retrieved_docs):
            chunk_trace = ChunkTrace(
                chunk_id=f"chunk_{i}",
                content=doc.page_content[:200],  # Truncate for storage
                source=doc.metadata.get("source", "unknown"),
                page_number=doc.metadata.get("page_number"),
                retrieval_score=doc.metadata.get("score", 0.0),
                selected=True,
                metadata=doc.metadata
            )
            self.current_trace.chunks.append(chunk_trace)
        
        # Update rerank scores
        if reranked_docs:
            for i, doc in enumerate(reranked_docs):
                for chunk_trace in self.current_trace.chunks:
                    if chunk_trace.content == doc.page_content[:200]:
                        chunk_trace.rerank_score = doc.metadata.get("rerank_score", 0.0)
                        chunk_trace.final_score = chunk_trace.rerank_score
        
        # Trace discarded chunks
        if discarded_docs:
            for i, doc in enumerate(discarded_docs):
                chunk_trace = ChunkTrace(
                    chunk_id=f"discarded_{i}",
                    content=doc.page_content[:200],
                    source=doc.metadata.get("source", "unknown"),
                    page_number=doc.metadata.get("page_number"),
                    retrieval_score=doc.metadata.get("score", 0.0),
                    selected=False,
                    discard_reason="low_relevance",
                    metadata=doc.metadata
                )
                self.current_trace.chunks.append(chunk_trace)
        
        logger.info(f"Traced {len(self.current_trace.chunks)} chunks")
    
    def set_query_info(self, query_type: str, retrieval_strategy: str) -> None:
        """Set query information.
        
        Args:
            query_type: Query type
            retrieval_strategy: Retrieval strategy used
        """
        if not self.current_trace:
            return
        
        self.current_trace.query_type = query_type
        self.current_trace.retrieval_strategy = retrieval_strategy
    
    def set_final_answer(self, answer: str, confidence: float) -> None:
        """Set final answer and confidence.
        
        Args:
            answer: Final generated answer
            confidence: Confidence score
        """
        if not self.current_trace:
            return
        
        self.current_trace.final_answer = answer
        self.current_trace.confidence_score = confidence
    
    def _save_trace(self, trace: PipelineTrace) -> None:
        """Save trace to file.
        
        Args:
            trace: Pipeline trace to save
        """
        trace_file = os.path.join(self.trace_dir, f"{trace.trace_id}.json")
        
        trace_data = {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "start_time": trace.start_time.isoformat(),
            "end_time": trace.end_time.isoformat() if trace.end_time else None,
            "total_duration_ms": trace.total_duration_ms,
            "query_type": trace.query_type,
            "retrieval_strategy": trace.retrieval_strategy,
            "final_answer": trace.final_answer,
            "confidence_score": trace.confidence_score,
            "success": trace.success,
            "error_message": trace.error_message,
            "metadata": trace.metadata,
            "stages": [
                {
                    "stage": stage.stage.value,
                    "start_time": stage.start_time.isoformat(),
                    "end_time": stage.end_time.isoformat() if stage.end_time else None,
                    "duration_ms": stage.duration_ms,
                    "input_data": stage.input_data,
                    "output_data": stage.output_data,
                    "success": stage.success,
                    "error_message": stage.error_message
                }
                for stage in trace.stages
            ],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "retrieval_score": chunk.retrieval_score,
                    "rerank_score": chunk.rerank_score,
                    "final_score": chunk.final_score,
                    "selected": chunk.selected,
                    "discard_reason": chunk.discard_reason,
                    "metadata": chunk.metadata
                }
                for chunk in trace.chunks
            ]
        }
        
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        logger.info(f"Saved trace to: {trace_file}")
    
    def get_trace(self, trace_id: str) -> Optional[PipelineTrace]:
        """Get a trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Pipeline trace if found, None otherwise
        """
        for trace in self.trace_history:
            if trace.trace_id == trace_id:
                return trace
        
        # Try loading from file
        trace_file = os.path.join(self.trace_dir, f"{trace_id}.json")
        if os.path.exists(trace_file):
            return self._load_trace(trace_file)
        
        return None
    
    def _load_trace(self, trace_file: str) -> Optional[PipelineTrace]:
        """Load trace from file.
        
        Args:
            trace_file: Path to trace file
            
        Returns:
            Pipeline trace if loaded successfully, None otherwise
        """
        try:
            with open(trace_file, 'r') as f:
                data = json.load(f)
            
            stages = [
                StageTrace(
                    stage=PipelineStage(stage_data["stage"]),
                    start_time=datetime.fromisoformat(stage_data["start_time"]),
                    end_time=datetime.fromisoformat(stage_data["end_time"]) if stage_data["end_time"] else None,
                    duration_ms=stage_data["duration_ms"],
                    input_data=stage_data["input_data"],
                    output_data=stage_data["output_data"],
                    success=stage_data["success"],
                    error_message=stage_data["error_message"]
                )
                for stage_data in data["stages"]
            ]
            
            chunks = [
                ChunkTrace(
                    chunk_id=chunk_data["chunk_id"],
                    content=chunk_data["content"],
                    source=chunk_data["source"],
                    page_number=chunk_data["page_number"],
                    retrieval_score=chunk_data["retrieval_score"],
                    rerank_score=chunk_data["rerank_score"],
                    final_score=chunk_data["final_score"],
                    selected=chunk_data["selected"],
                    discard_reason=chunk_data["discard_reason"],
                    metadata=chunk_data["metadata"]
                )
                for chunk_data in data["chunks"]
            ]
            
            trace = PipelineTrace(
                trace_id=data["trace_id"],
                query=data["query"],
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
                total_duration_ms=data["total_duration_ms"],
                stages=stages,
                chunks=chunks,
                final_answer=data["final_answer"],
                confidence_score=data["confidence_score"],
                query_type=data["query_type"],
                retrieval_strategy=data["retrieval_strategy"],
                metadata=data["metadata"],
                success=data["success"],
                error_message=data["error_message"]
            )
            
            return trace
        except Exception as e:
            logger.error(f"Failed to load trace: {e}")
            return None
    
    def get_recent_traces(self, n: int = 10) -> List[PipelineTrace]:
        """Get recent traces.
        
        Args:
            n: Number of recent traces to return
            
        Returns:
            List of recent traces
        """
        return self.trace_history[-n:]
    
    def get_aggregate_metrics(self, n: int = 100) -> Dict[str, Any]:
        """Get aggregate metrics from traces.
        
        Args:
            n: Number of recent traces to analyze
            
        Returns:
            Dictionary of aggregate metrics
        """
        recent_traces = self.trace_history[-n:]
        
        if not recent_traces:
            return {}
        
        total_duration = sum(trace.total_duration_ms for trace in recent_traces)
        successful_traces = sum(1 for trace in recent_traces if trace.success)
        
        # Stage duration metrics
        stage_durations = {}
        for trace in recent_traces:
            for stage in trace.stages:
                if stage.stage.value not in stage_durations:
                    stage_durations[stage.stage.value] = []
                stage_durations[stage.stage.value].append(stage.duration_ms)
        
        avg_stage_durations = {
            stage: sum(durations) / len(durations)
            for stage, durations in stage_durations.items()
        }
        
        return {
            "total_traces": len(recent_traces),
            "avg_duration_ms": total_duration / len(recent_traces),
            "success_rate": successful_traces / len(recent_traces),
            "avg_stage_durations": avg_stage_durations,
            "avg_confidence": sum(trace.confidence_score for trace in recent_traces) / len(recent_traces)
        }
    
    def clear_traces(self) -> None:
        """Clear all traces."""
        self.trace_history = []
        logger.info("Traces cleared")
