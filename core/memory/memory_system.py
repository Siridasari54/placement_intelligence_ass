"""AI Memory System for conversation memory, semantic cache, and FAQ memory."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Memory entry with metadata."""
    query: str
    answer: str
    timestamp: datetime
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime = None


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_turns: int = 10):
        """Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep
        """
        self.max_turns = max_turns
        self.conversation_history: List[Dict[str, Any]] = []
        
        logger.info(f"ConversationMemory initialized with max_turns={max_turns}")
    
    def add_turn(self, query: str, answer: str, metadata: Dict[str, Any] = None) -> None:
        """Add a conversation turn to memory.
        
        Args:
            query: User query
            answer: System answer
            metadata: Optional metadata
        """
        turn = {
            "query": query,
            "answer": answer,
            "timestamp": str(datetime.now()),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(turn)
        
        # Trim to max_turns
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]
        
        logger.info(f"Added conversation turn. Total turns: {len(self.conversation_history)}")
    
    def get_context(self, window_size: int = 3) -> str:
        """Get recent conversation context.
        
        Args:
            window_size: Number of recent turns to include
            
        Returns:
            Context string
        """
        recent_turns = self.conversation_history[-window_size:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"Q: {turn['query']}")
            context_parts.append(f"A: {turn['answer']}")
        
        return "\n".join(context_parts)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


class SemanticCache:
    """Semantic cache for similar query caching."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        cache_ttl: int = 3600
    ):
        """Initialize semantic cache.
        
        Args:
            model_name: Embedding model name
            similarity_threshold: Similarity threshold for cache hit
            cache_ttl: Cache time-to-live in seconds
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, MemoryEntry] = {}
        
        logger.info(f"SemanticCache initialized with threshold={similarity_threshold}")
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for similar query.
        
        Args:
            query: Query to look up
            
        Returns:
            Cached result if found, None otherwise
        """
        query_embedding = self.model.encode(query)
        
        # Check for similar queries in cache
        for key, entry in self.cache.items():
            # Check TTL
            if datetime.now() - entry.last_accessed > timedelta(seconds=self.cache_ttl):
                continue
            
            # Calculate similarity
            similarity = np.dot(query_embedding, entry.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
            )
            
            if similarity >= self.similarity_threshold:
                # Update access count and timestamp
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                logger.info(f"Cache hit with similarity {similarity:.2f}")
                return {
                    "answer": entry.answer,
                    "similarity": similarity,
                    "metadata": entry.metadata
                }
        
        logger.info("Cache miss")
        return None
    
    def set(self, query: str, answer: str, metadata: Dict[str, Any] = None) -> None:
        """Cache a query-answer pair.
        
        Args:
            query: Query text
            answer: Answer text
            metadata: Optional metadata
        """
        embedding = self.model.encode(query)
        
        entry = MemoryEntry(
            query=query,
            answer=answer,
            timestamp=datetime.now(),
            embedding=embedding.tolist(),
            metadata=metadata or {},
            access_count=1,
            last_accessed=datetime.now()
        )
        
        self.cache[query] = entry
        logger.info(f"Cached query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache = {}
        logger.info("Semantic cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        
        return {
            "cache_size": len(self.cache),
            "total_accesses": total_accesses,
            "avg_accesses": total_accesses / len(self.cache) if self.cache else 0
        }


class FAQMemory:
    """FAQ memory for frequently asked questions."""
    
    def __init__(self, faq_file: str = "data/faq.json"):
        """Initialize FAQ memory.
        
        Args:
            faq_file: Path to FAQ JSON file
        """
        self.faq_file = faq_file
        self.faqs: List[Dict[str, Any]] = []
        self._load_faqs()
        
        logger.info(f"FAQMemory initialized with {len(self.faqs)} FAQs")
    
    def _load_faqs(self) -> None:
        """Load FAQs from file."""
        if os.path.exists(self.faq_file):
            with open(self.faq_file, 'r') as f:
                self.faqs = json.load(f)
        else:
            self.faqs = []
    
    def _save_faqs(self) -> None:
        """Save FAQs to file."""
        os.makedirs(os.path.dirname(self.faq_file), exist_ok=True)
        with open(self.faq_file, 'w') as f:
            json.dump(self.faqs, f, indent=2)
    
    def add_faq(self, query: str, answer: str, frequency: int = 1) -> None:
        """Add or update FAQ entry.
        
        Args:
            query: FAQ question
            answer: FAQ answer
            frequency: Question frequency
        """
        # Check if FAQ already exists
        for faq in self.faqs:
            if faq["question"].lower() == query.lower():
                faq["frequency"] += frequency
                faq["last_updated"] = str(datetime.now())
                logger.info(f"Updated FAQ: {query[:50]}...")
                return
        
        # Add new FAQ
        self.faqs.append({
            "question": query,
            "answer": answer,
            "frequency": frequency,
            "created_at": str(datetime.now()),
            "last_updated": str(datetime.now())
        })
        
        self._save_faqs()
        logger.info(f"Added new FAQ: {query[:50]}...")
    
    def get_faq(self, query: str) -> Optional[Dict[str, Any]]:
        """Get FAQ answer for similar question.
        
        Args:
            query: Query to look up
            
        Returns:
            FAQ entry if found, None otherwise
        """
        query_lower = query.lower()
        
        for faq in self.faqs:
            if faq["question"].lower() == query_lower:
                logger.info(f"FAQ found: {query[:50]}...")
                return faq
        
        return None
    
    def get_top_faqs(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N FAQs by frequency.
        
        Args:
            n: Number of FAQs to return
            
        Returns:
            List of top FAQs
        """
        sorted_faqs = sorted(self.faqs, key=lambda x: x["frequency"], reverse=True)
        return sorted_faqs[:n]


class AIMemorySystem:
    """Unified AI memory system combining all memory types."""
    
    def __init__(self, max_conversation_turns: int = 10):
        """Initialize the AI memory system.
        
        Args:
            max_conversation_turns: Maximum conversation turns to keep
        """
        self.conversation_memory = ConversationMemory(max_turns=max_conversation_turns)
        self.semantic_cache = SemanticCache()
        self.faq_memory = FAQMemory()
        
        logger.info("AIMemorySystem initialized")
    
    def add_conversation_turn(
        self,
        query: str,
        answer: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add a conversation turn to memory.
        
        Args:
            query: User query
            answer: System answer
            metadata: Optional metadata
        """
        self.conversation_memory.add_turn(query, answer, metadata)
        
        # Cache if high confidence
        if metadata and metadata.get("confidence", 0) > 0.8:
            self.semantic_cache.set(query, answer, metadata)
        
        # Add to FAQ if frequently asked
        # (This would be triggered by user feedback in production)
    
    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available.
        
        Args:
            query: Query to look up
            
        Returns:
            Cached response if found, None otherwise
        """
        # Check semantic cache first
        cached = self.semantic_cache.get(query)
        if cached:
            return cached
        
        # Check FAQ memory
        faq = self.faq_memory.get_faq(query)
        if faq:
            return {
                "answer": faq["answer"],
                "source": "faq",
                "frequency": faq["frequency"]
            }
        
        return None
    
    def get_conversation_context(self, window_size: int = 3) -> str:
        """Get recent conversation context.
        
        Args:
            window_size: Number of recent turns
            
        Returns:
            Context string
        """
        return self.conversation_memory.get_context(window_size)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "conversation_turns": len(self.conversation_memory.get_history()),
            "cache_stats": self.semantic_cache.get_stats(),
            "faq_count": len(self.faq_memory.faqs)
        }
    
    def clear_all(self) -> None:
        """Clear all memory."""
        self.conversation_memory.clear()
        self.semantic_cache.clear()
        logger.info("All memory cleared")
