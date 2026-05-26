import os
from typing import Dict, Any

class PromptBuilder:
    def __init__(self, prompts_dir: str = "app/prompts"):
        self.prompts_dir = prompts_dir

    def _read_prompt_file(self, filename: str) -> str:
        """Reads a prompt template file from the prompts directory."""
        path = os.path.join(self.prompts_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # Inline fallback defaults if files are missing
            defaults = {
                "system_prompt.txt": "You are a Placement Intelligence Assistant. Use the context to answer.",
                "rag_prompt.txt": "Context:\n{context}\nQuestion: {question}\nCitations: {citations}",
                "fallback_prompt.txt": "Respond strictly with: 'I don't have enough information in the provided documents to answer this question.'",
                "conflict_prompt.txt": "State clearly that there are conflicting records. Present both: {context} for question: {question}",
                "temporal_prompt.txt": "Explain trend details using context: {context} for question: {question}"
            }
            return defaults.get(filename, "")

    def build_rag_prompt(self, question: str, context: str, citations: str) -> tuple:
        """Constructs system and user prompt for default RAG route."""
        system_prompt = self._read_prompt_file("system_prompt.txt")
        rag_template = self._read_prompt_file("rag_prompt.txt")
        
        user_prompt = rag_template.format(
            context=context,
            question=question,
            citations=citations
        )
        return user_prompt, system_prompt

    def build_fallback_prompt(self, question: str) -> tuple:
        """Constructs prompt for out-of-corpus queries."""
        system_prompt = self._read_prompt_file("system_prompt.txt")
        fallback_template = self._read_prompt_file("fallback_prompt.txt")
        
        user_prompt = fallback_template.format(question=question)
        return user_prompt, system_prompt

    def build_conflict_prompt(self, question: str, context: str) -> tuple:
        """Constructs prompt when conflict warning is triggered."""
        system_prompt = self._read_prompt_file("system_prompt.txt")
        conflict_template = self._read_prompt_file("conflict_prompt.txt")
        
        user_prompt = conflict_template.format(
            context=context,
            question=question
        )
        return user_prompt, system_prompt

    def build_temporal_prompt(self, question: str, context: str) -> tuple:
        """Constructs prompt for temporal queries."""
        system_prompt = self._read_prompt_file("system_prompt.txt")
        temporal_template = self._read_prompt_file("temporal_prompt.txt")
        
        user_prompt = temporal_template.format(
            context=context,
            question=question
        )
        return user_prompt, system_prompt
