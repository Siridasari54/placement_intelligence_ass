"""Query Planning Layer for intelligent query decomposition, routing, and planning."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query type classification for placement intelligence."""
    FACTUAL = "factual"  # Specific facts, numbers, dates
    ELIGIBILITY = "eligibility"  # Company eligibility criteria
    INTERNSHIP = "internship"  # Internship offers, stipend, duration
    STATISTICS = "statistics"  # Statistics, distributions, highest/lowest packages
    COMPARISON = "comparison"  # Compare multiple entities
    INTERVIEW = "interview"  # Interview experiences, process, rounds
    PLACEMENT_TREND = "placement_trend"  # Trends over time, yearly data
    PROCEDURAL = "procedural"  # How-to, processes
    DEFINITIONAL = "definitional"  # What is, explain concepts
    MULTI_HOP = "multi_hop"  # Requires multiple retrieval steps


@dataclass
class QueryPlan:
    """Plan for executing a query."""
    original_query: str
    query_type: QueryType
    sub_queries: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    retrieval_strategy: str = "balanced_hybrid"
    max_hops: int = 1
    requires_decomposition: bool = False
    confidence_threshold: float = 0.5


class QueryClassifier:
    """Classifies queries into types for appropriate routing."""
    
    def __init__(self):
        """Initialize the query classifier."""
        self.patterns = {
            QueryType.FACTUAL: [
                r'\b(what is|what are|how many|how much)\b',
                r'\b(number|count|amount|specific|exact)\b',
                r'\b(cgpa|package|salary|lpa|backlog)\b'
            ],
            QueryType.ELIGIBILITY: [
                r'\b(eligibility|eligible|requirement|criteria)\b',
                r'\b(qualify|qualification|condition)\b',
                r'\b(minimum|maximum|needed)\b',
                r'\b(cgpa|backlog|percentage)\b'
            ],
            QueryType.INTERNSHIP: [
                r'\b(internship|intern|stipend|duration)\b',
                r'\b(offer|placement|summer|winter)\b',
                r'\b(internship program|internship offer)\b'
            ],
            QueryType.STATISTICS: [
                r'\b(statistics|distribution|average|median|mean)\b',
                r'\b(percentage|ratio|proportion)\b',
                r'\b(hired|placed|selected)\b',
                r'\b(highest|lowest|maximum|minimum|top)\b',
                r'\b(package|salary|lpa)\b'
            ],
            QueryType.COMPARISON: [
                r'\b(compare|difference|versus|vs|better|worse)\b',
                r'\b(and|both|between)\b',
                r'\b(amazon|google|microsoft|tcs|infosys)\b.*\b(and|vs|versus)\b'
            ],
            QueryType.INTERVIEW: [
                r'\b(interview|selection|recruitment)\b',
                r'\b(round|stage|phase|experience)\b',
                r'\b(technical|hr|coding)\b',
                r'\b(interview process|interview experience)\b'
            ],
            QueryType.PLACEMENT_TREND: [
                r'\b(trend|change|over time|history|evolution)\b',
                r'\b(increasing|decreasing|growth|decline)\b',
                r'\b(year|yearly|annual|seasonal)\b',
                r'\b(placement statistics|placement data)\b'
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|how do|process|steps)\b',
                r'\b(interview|selection|recruitment)\b',
                r'\b(round|stage|phase)\b'
            ],
            QueryType.DEFINITIONAL: [
                r'\b(what is|what does|explain|describe)\b',
                r'\b(meaning|definition|concept)\b'
            ]
        }
        
        logger.info("QueryClassifier initialized with enhanced patterns")
    
    def classify(self, query: str) -> QueryType:
        """Classify query into type.
        
        Args:
            query: User query
            
        Returns:
            QueryType classification
        """
        query_lower = query.lower()
        scores = {}
        
        for query_type, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 0.3
            scores[query_type] = min(score, 1.0)
        
        # Select type with highest score
        best_type = max(scores.items(), key=lambda x: x[1])[0]
        
        # Check for multi-hop indicators
        if self._is_multi_hop(query):
            return QueryType.MULTI_HOP
        
        logger.info(f"Classified query as: {best_type.value}")
        return best_type
    
    def _is_multi_hop(self, query: str) -> bool:
        """Check if query requires multi-hop retrieval.
        
        Args:
            query: Query text
            
        Returns:
            True if multi-hop, False otherwise
        """
        multi_hop_indicators = [
            r'\b(and then|after that|followed by)\b',
            r'\b(compare.*and.*compare)\b',
            r'\b(based on|depending on)\b.*\b(then|also)\b'
        ]
        
        query_lower = query.lower()
        for pattern in multi_hop_indicators:
            if re.search(pattern, query_lower):
                return True
        
        # Check for multiple company names
        companies = ['amazon', 'google', 'microsoft', 'tcs', 'infosys', 'facebook', 'meta']
        company_count = sum(1 for company in companies if company in query_lower)
        if company_count >= 2:
            return True
        
        return False


class QueryDecomposer:
    """Decomposes complex queries into sub-queries."""
    
    def __init__(self):
        """Initialize the query decomposer."""
        logger.info("QueryDecomposer initialized")
    
    def decompose(self, query: str, query_type: QueryType) -> List[str]:
        """Decompose query into sub-queries.
        
        Args:
            query: Original query
            query_type: Query type
            
        Returns:
            List of sub-queries
        """
        if query_type == QueryType.MULTI_HOP:
            return self._decompose_multi_hop(query)
        elif query_type == QueryType.COMPARISON:
            return self._decompose_comparison(query)
        else:
            return [query]
    
    def _decompose_multi_hop(self, query: str) -> List[str]:
        """Decompose multi-hop query.
        
        Args:
            query: Multi-hop query
            
        Returns:
            List of sub-queries
        """
        sub_queries = []
        
        # Split by common conjunctions
        parts = re.split(r'\b(and|then|after that)\b', query, flags=re.IGNORECASE)
        sub_queries = [part.strip() for part in parts if part.strip()]
        
        # If no split, try to extract entities
        if len(sub_queries) == 1:
            companies = self._extract_companies(query)
            if len(companies) >= 2:
                for company in companies:
                    sub_queries.append(query.replace(company, "").strip() + f" for {company}")
        
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
        return sub_queries
    
    def _decompose_comparison(self, query: str) -> List[str]:
        """Decompose comparison query.
        
        Args:
            query: Comparison query
            
        Returns:
            List of sub-queries
        """
        companies = self._extract_companies(query)
        
        if len(companies) >= 2:
            base_query = re.sub(r'\b(compare|vs|versus|difference)\b.*', '', query, flags=re.IGNORECASE).strip()
            sub_queries = [f"{base_query} for {company}" for company in companies]
            logger.info(f"Decomposed comparison into {len(sub_queries)} sub-queries")
            return sub_queries
        
        return [query]
    
    def _extract_companies(self, query: str) -> List[str]:
        """Extract company names from query.
        
        Args:
            query: Query text
            
        Returns:
            List of company names
        """
        companies = ['amazon', 'google', 'microsoft', 'tcs', 'infosys', 'facebook', 'meta', 'apple', 'netflix']
        query_lower = query.lower()
        return [company for company in companies if company in query_lower]


class QueryRewriter:
    """Rewrites queries for better retrieval."""
    
    def __init__(self):
        """Initialize the query rewriter."""
        self.rewrite_rules = {
            # Expand abbreviations
            r'\bcgpa\b': 'cumulative grade point average cgpa',
            r'\blpa\b': 'lakhs per annum lpa package salary',
            r'\bsde\b': 'software development engineer sde',
            r'\bhr\b': 'human resources hr',
            
            # Add contextual terms
            r'\b(requirement|criteria)\b': r'\1 eligibility qualification',
            r'\b(package|salary)\b': r'\1 compensation pay',
            r'\b(interview|selection)\b': r'\1 recruitment process'
        }
        
        logger.info("QueryRewriter initialized")
    
    def rewrite(self, query: str) -> str:
        """Rewrite query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        rewritten = query
        
        for pattern, replacement in self.rewrite_rules.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        logger.info(f"Rewritten query: {rewritten}")
        return rewritten


class RetrievalRouter:
    """Routes queries to appropriate retrieval strategies."""
    
    def __init__(self, adaptive_strategy):
        """Initialize the retrieval router.
        
        Args:
            adaptive_strategy: AdaptiveRetrievalStrategy instance
        """
        self.adaptive_strategy = adaptive_strategy
        self.type_to_strategy = {
            QueryType.FACTUAL: "keyword_heavy",
            QueryType.ELIGIBILITY: "metadata_first",
            QueryType.INTERNSHIP: "metadata_first",
            QueryType.STATISTICS: "keyword_heavy",
            QueryType.COMPARISON: "balanced_hybrid",
            QueryType.INTERVIEW: "semantic_heavy",
            QueryType.PLACEMENT_TREND: "balanced_hybrid",
            QueryType.PROCEDURAL: "semantic_heavy",
            QueryType.DEFINITIONAL: "semantic_heavy",
            QueryType.MULTI_HOP: "multi_hop"
        }
        
        logger.info("RetrievalRouter initialized with enhanced type mapping")
    
    def route(self, query: str, query_type: QueryType, metadata_filters: Dict[str, Any] = None) -> str:
        """Route query to appropriate retrieval strategy.
        
        Args:
            query: User query
            query_type: Query type
            metadata_filters: Optional metadata filters
            
        Returns:
            Retrieval strategy name
        """
        # Get base strategy from type
        base_strategy = self.type_to_strategy.get(query_type, "balanced_hybrid")
        
        # Use adaptive strategy for fine-tuning
        mode = self.adaptive_strategy.select_mode(query, metadata_filters)
        
        # Combine type-based and adaptive strategies
        if query_type == QueryType.ELIGIBILITY and metadata_filters:
            return "metadata_first"
        elif query_type == QueryType.MULTI_HOP:
            return "multi_hop"
        else:
            return mode.value
        
        logger.info(f"Routed to strategy: {base_strategy}")
        return base_strategy


class QueryPlanner:
    """Main query planner that coordinates classification, decomposition, and routing."""
    
    def __init__(self, adaptive_strategy=None):
        """Initialize the query planner.
        
        Args:
            adaptive_strategy: Optional AdaptiveRetrievalStrategy instance
        """
        self.classifier = QueryClassifier()
        self.decomposer = QueryDecomposer()
        self.rewriter = QueryRewriter()
        self.router = RetrievalRouter(adaptive_strategy) if adaptive_strategy else None
        
        logger.info("QueryPlanner initialized")
    
    def plan(self, query: str, metadata_filters: Dict[str, Any] = None) -> QueryPlan:
        """Create execution plan for query.
        
        Args:
            query: User query
            metadata_filters: Optional metadata filters
            
        Returns:
            QueryPlan object
        """
        logger.info(f"Planning query: {query}")
        
        # Classify query
        query_type = self.classifier.classify(query)
        
        # Rewrite query
        rewritten_query = self.rewriter.rewrite(query)
        
        # Decompose if needed
        sub_queries = self.decomposer.decompose(rewritten_query, query_type)
        requires_decomposition = len(sub_queries) > 1
        
        # Determine retrieval strategy
        if self.router:
            retrieval_strategy = self.router.route(rewritten_query, query_type, metadata_filters)
        else:
            retrieval_strategy = "balanced_hybrid"
        
        # Set max hops based on query type
        max_hops = 3 if query_type == QueryType.MULTI_HOP else 1
        
        # Set confidence threshold based on query type
        confidence_threshold = {
            QueryType.FACTUAL: 0.6,
            QueryType.ELIGIBILITY: 0.5,
            QueryType.COMPARISON: 0.5,
            QueryType.PLACEMENT_TREND: 0.5,
            QueryType.STATISTICS: 0.6,
            QueryType.PROCEDURAL: 0.5,
            QueryType.DEFINITIONAL: 0.4,
            QueryType.MULTI_HOP: 0.4
        }.get(query_type, 0.5)
        
        plan = QueryPlan(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            metadata_filters=metadata_filters or {},
            retrieval_strategy=retrieval_strategy,
            max_hops=max_hops,
            requires_decomposition=requires_decomposition,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Query plan created: {query_type.value}, strategy: {retrieval_strategy}")
        return plan
    
    def execute_plan(
        self,
        plan: QueryPlan,
        retriever,
        k: int = 5
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Execute query plan.
        
        Args:
            plan: QueryPlan object
            retriever: Retriever instance
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (retrieved documents, execution metadata)
        """
        logger.info(f"Executing plan for: {plan.original_query}")
        
        all_docs = []
        execution_metadata = {
            "query_type": plan.query_type.value,
            "retrieval_strategy": plan.retrieval_strategy,
            "sub_queries_executed": len(plan.sub_queries),
            "total_retrieved": 0,
            "sub_query_results": []
        }
        
        # Execute sub-queries
        for sub_query in plan.sub_queries:
            docs, metadata = retriever.retrieve(
                sub_query,
                k=k,
                metadata_filters=plan.metadata_filters,
                strategy=plan.retrieval_strategy
            )
            all_docs.extend(docs)
            execution_metadata["sub_query_results"].append({
                "sub_query": sub_query,
                "docs_retrieved": len(docs),
                "metadata": metadata
            })
        
        # Deduplicate documents
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = str(doc)
            if doc_id not in seen:
                unique_docs.append(doc)
                seen.add(doc_id)
        
        execution_metadata["total_retrieved"] = len(unique_docs)
        
        logger.info(f"Plan execution complete: {len(unique_docs)} documents retrieved")
        return unique_docs, execution_metadata
