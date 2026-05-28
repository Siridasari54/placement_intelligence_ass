"""Automated evaluator with visual summary for RAG system performance."""

from typing import List, Dict, Any, Tuple
import json
import os
from datetime import datetime
from evaluation.queries import EVALUATION_QUERIES
from evaluation.metrics import Evaluator
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class AutomatedEvaluator:
    """Automated evaluator for running full evaluation pipeline."""
    
    def __init__(self, rag_pipeline):
        """Initialize the automated evaluator.
        
        Args:
            rag_pipeline: RAG pipeline instance to evaluate
        """
        self.rag_pipeline = rag_pipeline
        self.evaluator = Evaluator()
        self.results = []
        logger.info("AutomatedEvaluator initialized")
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation pipeline on all queries.
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation on {len(EVALUATION_QUERIES)} queries")
        
        all_results = []
        
        for i, query in enumerate(EVALUATION_QUERIES):
            logger.info(f"Evaluating query {i+1}/{len(EVALUATION_QUERIES)}: {query}")
            
            try:
                # Execute query through RAG pipeline
                result = self.rag_pipeline.query(query)
                
                # Evaluate result (placeholder - would need ground truth)
                metrics = {
                    "query": query,
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0.0),
                    "sources_count": len(result.get("sources", [])),
                    "conflicts": result.get("conflicts", 0)
                }
                
                all_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating query {query}: {e}")
                all_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregates(all_results)
        
        # Save results
        self._save_results(all_results, aggregate_metrics)
        
        logger.info("Evaluation complete")
        return {
            "individual_results": all_results,
            "aggregate_metrics": aggregate_metrics
        }
    
    def _calculate_aggregates(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results.
        
        Args:
            results: Individual query results
            
        Returns:
            Dictionary of aggregate metrics
        """
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {}
        
        avg_confidence = sum(r.get("confidence", 0) for r in valid_results) / len(valid_results)
        avg_sources = sum(r.get("sources_count", 0) for r in valid_results) / len(valid_results)
        total_conflicts = sum(r.get("conflicts", 0) for r in valid_results)
        
        return {
            "total_queries": len(results),
            "successful_queries": len(valid_results),
            "failed_queries": len(results) - len(valid_results),
            "avg_confidence": avg_confidence,
            "avg_sources": avg_sources,
            "total_conflicts": total_conflicts
        }
    
    def _save_results(self, results: List[Dict[str, Any]], aggregates: Dict[str, float]) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Individual query results
            aggregates: Aggregate metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join(settings.evaluation.metrics_output_dir, filename)
        
        output = {
            "timestamp": timestamp,
            "individual_results": results,
            "aggregate_metrics": aggregates
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
