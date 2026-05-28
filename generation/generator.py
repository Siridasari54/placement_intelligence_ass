"""Answer generator with Groq LLM integration and hallucination reduction."""

from typing import List, Tuple
from langchain_core.documents import Document
from groq import Groq
from core.interfaces import IGenerator
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class GroqGenerator(IGenerator):
    """Groq LLM-based answer generator with hallucination reduction."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Groq generator.
        
        Args:
            api_key: Groq API key (defaults to settings)
        """
        self.api_key = api_key or settings.groq_api_key
        self.client = Groq(api_key=self.api_key)
        self.model = settings.generation.model
        logger.info(f"GroqGenerator initialized with model: {self.model}")
    
    def generate(self, query: str, context: List[Document]) -> str:
        """Generate an answer based on query and context.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            Generated answer string
        """
        logger.info("Generating answer")
        
        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}"
            for i, doc in enumerate(context)
        ])
        
        # Build prompt
        prompt = f"""You are a Placement Intelligence Assistant. Answer the following question based ONLY on the provided context.

Context:
{context_text}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain the answer, state that clearly
- Provide specific numbers, dates, and company names when available
- Be concise but thorough
- Cite sources using [Source X] notation

Answer:"""
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful placement intelligence assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.generation.temperature,
                max_tokens=settings.generation.max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."
