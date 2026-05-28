"""Evaluation metrics for retrieval quality and answer consistency."""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    @staticmethod
    def precision_at_k(retrieved: List[Document], expected: List[str], k: int) -> float:
        """Calculate precision at k.
        
        Args:
            retrieved: Retrieved documents
            expected: Expected document IDs
            k: Number of top documents to consider
            
        Returns:
            Precision score
        """
        if not retrieved or not expected:
            return 0.0
        
        retrieved_ids = [doc.metadata.get("id", "") for doc in retrieved[:k]]
        relevant = sum(1 for doc_id in retrieved_ids if doc_id in expected)
        
        return relevant / k
    
    @staticmethod
    def recall_at_k(retrieved: List[Document], expected: List[str], k: int) -> float:
        """Calculate recall at k.
        
        Args:
            retrieved: Retrieved documents
            expected: Expected document IDs
            k: Number of top documents to consider
            
        Returns:
            Recall score
        """
        if not expected:
            return 0.0
        
        retrieved_ids = [doc.metadata.get("id", "") for doc in retrieved[:k]]
        relevant = sum(1 for doc_id in retrieved_ids if doc_id in expected)
        
        return relevant / len(expected)
    
    @staticmethod
    def mrr(retrieved: List[Document], expected: List[str]) -> float:
        """Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved: Retrieved documents
            expected: Expected document IDs
            
        Returns:
            MRR score
        """
        if not retrieved or not expected:
            return 0.0
        
        for i, doc in enumerate(retrieved):
            if doc.metadata.get("id", "") in expected:
                return 1.0 / (i + 1)
        
        return 0.0


class AnswerMetrics:
    """Metrics for evaluating answer quality."""
    
    @staticmethod
    def exact_match(generated: str, expected: str) -> float:
        """Calculate exact match score.
        
        Args:
            generated: Generated answer
            expected: Expected answer
            
        Returns:
            Exact match score (0 or 1)
        """
        return 1.0 if generated.strip() == expected.strip() else 0.0
    
    @staticmethod
    def keyword_overlap(generated: str, expected: str) -> float:
        """Calculate keyword overlap score.
        
        Args:
            generated: Generated answer
            expected: Expected answer
            
        Returns:
            Keyword overlap score
        """
        generated_words = set(generated.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(generated_words & expected_words)
        return overlap / len(expected_words)
    
    @staticmethod
    def citation_coverage(generated: str, context: List[Document]) -> float:
        """Calculate citation coverage in generated answer.
        
        Args:
            generated: Generated answer
            context: Context documents
            
        Returns:
            Citation coverage score
        """
        if not context:
            return 0.0
        
        # Check if answer cites sources
        has_citations = "[Source" in generated or "[source" in generated
        return 1.0 if has_citations else 0.0
    
    @staticmethod
    def citation_accuracy(generated: str, context: List[Document]) -> float:
        """Calculate citation accuracy - whether citations match actual sources.
        
        Args:
            generated: Generated answer
            context: Context documents
            
        Returns:
            Citation accuracy score between 0 and 1
        """
        if not context:
            return 0.0
        
        # Extract citation numbers from answer
        import re
        citations = re.findall(r'\[Source (\d+)\]', generated)
        
        if not citations:
            return 0.0
        
        # Check if citations are valid (within context range)
        valid_citations = 0
        for citation in citations:
            try:
                citation_num = int(citation)
                if 1 <= citation_num <= len(context):
                    valid_citations += 1
            except ValueError:
                continue
        
        return valid_citations / len(citations) if citations else 0.0
    
    @staticmethod
    def hallucination_score(generated: str, context: List[Document]) -> float:
        """Calculate hallucination score - lower is better (0 = no hallucination).
        
        Args:
            generated: Generated answer
            context: Context documents
            
        Returns:
            Hallucination score between 0 and 1 (0 = no hallucination, 1 = high hallucination)
        """
        if not context:
            return 1.0  # High hallucination risk if no context
        
        # Extract key entities from context
        context_text = " ".join([doc.page_content for doc in context])
        context_words = set(context_text.lower().split())
        
        # Extract words from generated answer
        generated_words = set(generated.lower().split())
        
        # Calculate overlap
        overlap = len(generated_words & context_words)
        
        if not generated_words:
            return 1.0
        
        # High overlap = low hallucination
        overlap_ratio = overlap / len(generated_words)
        hallucination_score = 1.0 - overlap_ratio
        
        return hallucination_score
    
    @staticmethod
    def faithfulness_score(generated: str, context: List[Document]) -> float:
        """Calculate faithfulness score - how well answer is grounded in context.
        
        Args:
            generated: Generated answer
            context: Context documents
            
        Returns:
            Faithfulness score between 0 and 1 (1 = fully faithful)
        """
        if not context:
            return 0.0
        
        # Check for grounding indicators
        grounding_indicators = [
            "based on the documents",
            "according to the",
            "the documents state",
            "as mentioned in",
            "source"
        ]
        
        grounding_count = sum(1 for indicator in grounding_indicators if indicator in generated.lower())
        
        # Calculate citation accuracy
        citation_acc = AnswerMetrics.citation_accuracy(generated, context)
        
        # Calculate hallucination (inverse)
        hallucination = AnswerMetrics.hallucination_score(generated, context)
        
        # Combine metrics for faithfulness
        faithfulness = (grounding_count / len(grounding_indicators) * 0.3) + (citation_acc * 0.4) + ((1 - hallucination) * 0.3)
        
        return min(1.0, max(0.0, faithfulness))


class Evaluator:
    """Comprehensive evaluator for RAG system performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerMetrics()
        logger.info("Evaluator initialized")
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved: List[Document],
        expected: List[str]
    ) -> Dict[str, float]:
        """Evaluate retrieval quality.
        
        Args:
            query: Query text
            retrieved: Retrieved documents
            expected: Expected document IDs
            
        Returns:
            Dictionary of metric scores
        """
        logger.info(f"Evaluating retrieval for query: {query}")
        
        return {
            "precision@5": self.retrieval_metrics.precision_at_k(retrieved, expected, 5),
            "precision@10": self.retrieval_metrics.precision_at_k(retrieved, expected, 10),
            "recall@5": self.retrieval_metrics.recall_at_k(retrieved, expected, 5),
            "recall@10": self.retrieval_metrics.recall_at_k(retrieved, expected, 10),
            "mrr": self.retrieval_metrics.mrr(retrieved, expected)
        }
    
    def evaluate_answer(
        self,
        query: str,
        answer: str,
        expected: str,
        context: List[Document] = None
    ) -> Dict[str, float]:
        """Evaluate answer quality with comprehensive metrics.
        
        Args:
            query: Query text
            answer: Generated answer
            expected: Expected answer
            context: Context documents
            
        Returns:
            Dictionary of metric scores
        """
        logger.info(f"Evaluating answer for query: {query}")
        
        metrics = {
            "exact_match": self.answer_metrics.exact_match(answer, expected),
            "keyword_overlap": self.answer_metrics.keyword_overlap(answer, expected)
        }
        
        if context:
            metrics["citation_coverage"] = self.answer_metrics.citation_coverage(answer, context)
            metrics["citation_accuracy"] = self.answer_metrics.citation_accuracy(answer, context)
            metrics["hallucination_score"] = self.answer_metrics.hallucination_score(answer, context)
            metrics["faithfulness_score"] = self.answer_metrics.faithfulness_score(answer, context)
        
        return metrics
