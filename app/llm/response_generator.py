from typing import List, Dict, Any
from app.llm.groq_llm import GroqLLM
from app.llm.prompt_builder import PromptBuilder
from app.utils.citation_builder import build_citations_list
from app.utils.logger import logger

class ResponseGenerator:
    def __init__(self):
        self.llm = GroqLLM()
        self.prompt_builder = PromptBuilder()

    def generate_response(self, question: str, chunks: List[Dict[str, Any]], route: str) -> str:
        """Assembles prompt according to query route and fetches LLM response."""
        logger.info(f"Generating response for route: {route}...")
        
        # 1. Fallback Route
        if route == "fallback":
            user_prompt, system_prompt = self.prompt_builder.build_fallback_prompt(question)
            return self.llm.generate(user_prompt, system_message=system_prompt)
            
        # 2. Extract context text and citations
        context = "\n\n".join([chunk.get("text", "") for chunk in chunks])
        citations_list = build_citations_list(chunks)
        citations_str = "\n".join(citations_list) if citations_list else "No source files cited."
        
        # 3. Route specific prompt assemblies
        if route == "conflict":
            user_prompt, system_prompt = self.prompt_builder.build_conflict_prompt(question, context)
        elif route == "temporal":
            user_prompt, system_prompt = self.prompt_builder.build_temporal_prompt(question, context)
        else:
            # Standard, Eligibility, Hiring, Multi-hop
            user_prompt, system_prompt = self.prompt_builder.build_rag_prompt(question, context, citations_str)
            
        return self.llm.generate(user_prompt, system_message=system_prompt)
