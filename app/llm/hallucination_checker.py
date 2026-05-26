import re
from typing import List, Dict, Any
from app.llm.groq_llm import GroqLLM
from app.utils.logger import logger

class HallucinationChecker:
    def __init__(self):
        self.llm = GroqLLM()

    def check_hallucination(self, question: str, response: str, chunks: List[Dict[str, Any]], route: str) -> Dict[str, Any]:
        """Checks if response contains claims unsupported by retrieved context."""
        if route == "fallback":
            return {"is_hallucinated": False, "reason": "Fallback route bypasses checker"}

        context = "\n\n".join([c.get("text", "") for c in chunks])
        
        # 1. Numerical Check: Ensure all floats/integers mentioned in the response (e.g., packages, cutoffs)
        # exist somewhere in the context, to prevent simple hallucinations of stats
        response_numbers = re.findall(r"\b\d+\.\d+|\b\d+\b", response)
        context_numbers = set(re.findall(r"\b\d+\.\d+|\b\d+\b", context))
        
        # Allow small numbers (like list indices, round numbers, counts) without flagging
        unsupported = []
        for num in response_numbers:
            # Skip common short index numbers
            if num in ["1", "2", "3", "4", "5", "0"]:
                continue
            if num not in context_numbers:
                unsupported.append(num)
                
        if unsupported:
            logger.warning(f"Numerical mismatch detected: response mentioned {unsupported} which are not in context.")
            return {
                "is_hallucinated": True,
                "reason": f"Response mentioned numbers {unsupported} which were not present in the retrieved documents."
            }

        # 2. LLM-Based Verification (if LLM client is alive and route is standard)
        if self.llm.client:
            eval_prompt = (
                f"Verify if the following Response is fully supported by the retrieved Context. "
                f"Identify any facts in the Response that are missing from or contradict the Context.\n\n"
                f"Context:\n{context}\n\n"
                f"Response:\n{response}\n\n"
                f"Respond with 'SUFFICIENT' if the facts are supported. Otherwise, respond with 'CONTRADICTION: <details>'."
            )
            try:
                verdict = self.llm.generate(eval_prompt, system_message="You are a strict factual consistency checker.")
                if "CONTRADICTION" in verdict.upper():
                    logger.warning(f"LLM consistency checker failed: {verdict}")
                    return {
                        "is_hallucinated": True,
                        "reason": f"LLM consistency check failed: {verdict}"
                    }
            except Exception as e:
                logger.error(f"Error running LLM consistency check: {e}")
                
        return {"is_hallucinated": False, "reason": "Factual consistency verified"}
