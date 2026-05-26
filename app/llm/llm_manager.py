from typing import List, Dict, Any, Tuple
from app.llm.response_generator import ResponseGenerator
from app.llm.hallucination_checker import HallucinationChecker

class LLMManager:
    _instance = None
    _generator: ResponseGenerator = None
    _checker: HallucinationChecker = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._generator = ResponseGenerator()
            cls._instance._checker = HallucinationChecker()
        return cls._instance

    def generate(self, question: str, chunks: List[Dict[str, Any]], route: str) -> Tuple[str, Dict[str, Any]]:
        """Generates response and returns it along with hallucination evaluation check."""
        response = self._generator.generate_response(question, chunks, route)
        eval_result = self._checker.check_hallucination(question, response, chunks, route)
        
        # If hallucination detected, we can automatically append a system correction warning or regenerate
        if eval_result["is_hallucinated"]:
            response = (
                f"{response}\n\n"
                f"> [!WARNING]\n"
                f"> **Factual Warning**: {eval_result['reason']}. Please cross-verify."
            )
            
        return response, eval_result

llm_manager = LLMManager()
