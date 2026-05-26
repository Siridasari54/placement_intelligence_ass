import time
from typing import Dict, Any
from app.utils.logger import logger
from app.cache.cache_manager import cache_manager
from app.agents.workflow import graph_app

class RAGService:
    def __init__(self):
        self.cache = cache_manager.get_cache()

    def query(self, query_text: str) -> Dict[str, Any]:
        """Processes query text through semantic cache and LangGraph agent workflow."""
        start_time = time.time()
        
        # 1. Semantic Cache Check
        if self.cache:
            try:
                cached_ans, meta = self.cache.get(query_text)
                if cached_ans:
                    latency = time.time() - start_time
                    return {
                        "query": query_text,
                        "response": cached_ans,
                        "route": "cached",
                        "citations": ["Cached Answer - SQLite DB Store"],
                        "is_hallucinated": False,
                        "cache_hit": True,
                        "metrics": {
                            "latency_seconds": round(latency, 4),
                            "cache_score": meta.get("score", 1.0),
                            "overall_quality": 5.0
                        }
                    }
            except Exception as e:
                logger.error(f"Error checking cache: {e}")
                
        # 2. Invoke LangGraph workflow
        logger.info(f"Invoking agent graph for query: '{query_text}'")
        inputs = {
            "query": query_text,
            "route": "",
            "retrieved_chunks": [],
            "response": "",
            "citations": [],
            "is_hallucinated": False,
            "validation_reason": "",
            "evaluation_metrics": {}
        }
        
        try:
            outputs = graph_app.invoke(inputs)
            latency = time.time() - start_time
            
            response = outputs["response"]
            route = outputs["route"]
            is_hall = outputs["is_hallucinated"]
            metrics = outputs["evaluation_metrics"]
            metrics["latency_seconds"] = round(latency, 4)
            
            # 3. Update Semantic Cache if response is valid (not hallucinated, not fallback)
            if self.cache and not is_hall and route not in ["fallback", "edge_case_cgpa"]:
                try:
                    self.cache.set(query_text, response)
                except Exception as cache_err:
                    logger.error(f"Error writing to cache: {cache_err}")
                    
            return {
                "query": query_text,
                "response": response,
                "route": route,
                "citations": outputs["citations"],
                "is_hallucinated": is_hall,
                "cache_hit": False,
                "metrics": metrics
            }
            
        except Exception as graph_err:
            logger.error(f"Error executing agent workflow graph: {graph_err}")
            latency = time.time() - start_time
            return {
                "query": query_text,
                "response": "An internal error occurred while generating the answer. Please try again.",
                "route": "error",
                "citations": [],
                "is_hallucinated": False,
                "cache_hit": False,
                "metrics": {"latency_seconds": round(latency, 4), "overall_quality": 0.0}
            }

rag_service = RAGService()
