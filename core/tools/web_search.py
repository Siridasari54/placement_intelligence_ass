"""Web Search tool for fetching live information outside the placement dataset using Tavily Search."""

import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq
from config.settings import settings

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Fetches live information from the web via Tavily Search when queries are out-of-corpus.
    
    If Tavily is offline or the API key is invalid, it gracefully falls back to generating
    a direct answer using the LLM's internal knowledge base.
    """
    
    def __init__(self):
        # 1. Explicitly load .env to ensure environment variables are populated
        load_dotenv()
        
        # 2. Retrieve TAVILY_API_KEY from environment
        raw_key = os.getenv("TAVILY_API_KEY", "")
        
        # 5. Clean for extra spaces, enclosing quotes, or formatting issues
        self.tavily_api_key = raw_key.strip().strip("'").strip('"') if raw_key else ""
        
        self.groq_api_key = settings.groq_api_key
        self.model = settings.generation.model
        
        self.tavily_client = None
        self.groq_client = None
        
        # 3. Securely log loaded API key status showing only first 5 characters
        if self.tavily_api_key:
            masked_key = self.tavily_api_key[:5] + "*" * max(0, len(self.tavily_api_key) - 5)
            logger.info(f"TAVILY_API_KEY successfully read from environment. Masked key: {masked_key}")
        else:
            # 6. Log clear error message if the key is missing
            logger.error("TAVILY_API_KEY is missing from environment variables! Please set it in your .env file.")
            
        # 4. Verify Tavily client is initialized correctly
        if self.tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
                logger.info("Tavily Search client successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily Search client: {e}")
                
        # Initialize Groq client
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("Groq client successfully initialized in WebSearchTool.")
            except Exception as e:
                logger.error(f"Error initializing Groq client in WebSearchTool: {e}")
        else:
            logger.warning("Groq API key is missing. Answer generation fallback is unavailable.")
            
        logger.info("WebSearchTool initialization complete.")
        
    def execute(self, query: str) -> str:
        """Execute web search for the query using Tavily and generate a direct answer.
        
        If Tavily fails, gracefully fall back to generating a direct answer using Groq internal knowledge.
        
        Args:
            query: User search query
            
        Returns:
            Natural-language answer with source citations at the bottom.
        """
        logger.info(f"WebSearchTool executing query: {query}")
        
        tavily_success = False
        results = []
        tavily_direct_answer = ""
        
        # 7. Add graceful error handling for Tavily API failures
        if self.tavily_client:
            try:
                logger.info("Sending search request to Tavily...")
                search_response = self.tavily_client.search(
                    query=query,
                    max_results=3,
                    search_depth="basic",
                    include_answer=True
                )
                
                raw_results = search_response.get("results", [])
                tavily_direct_answer = search_response.get("answer", "")
                
                # Format into clean structured results compatible with RAG pipeline
                for r in raw_results:
                    results.append({
                        "title": r.get("title", "No Title"),
                        "href": r.get("url", "#"),
                        "body": r.get("content", "")
                    })
                    
                if results:
                    tavily_success = True
                    logger.info(f"Successfully retrieved {len(results)} search results from Tavily.")
            except Exception as e:
                logger.error(f"Tavily API search failed: {e}. Gracefully falling back to LLM internal knowledge.", exc_info=True)
        else:
            logger.warning("Tavily client is not initialized. Using LLM internal knowledge.")
            
        # 9. Ensure the chatbot returns a direct answer generated from Tavily results/LLM instead of showing raw errors.
        if not tavily_success:
            logger.info("Synthesizing direct answer using Groq LLM internal knowledge as fallback...")
            
            if not self.groq_client:
                return "I apologize, but both the search service and answer generation services are currently offline."
                
            prompt = f"""You are a precise and helpful college placement assistant.
Please provide a concise, direct, and factual answer to the user's question using your internal knowledge.

User Question: "{query}"

Instructions:
1. Provide a concise, direct, and factual answer (usually 1-3 sentences).
2. Start your answer immediately with the information - do NOT use introductory phrases like "Based on...", "According to...", etc.
3. Keep the answer professional and completely accurate.

Concise Direct Answer:"""

            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise assistant that provides direct, concise factual answers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=256
                )
                
                llm_answer = response.choices[0].message.content.strip()
                llm_answer = llm_answer.replace("---", "").replace("***", "").strip()
                
                # Format direct answer with clean citation footnote indicating search fallback
                formatted_output = f"{llm_answer}\n\n"
                formatted_output += "---\n"
                formatted_output += "**Source:**\n"
                formatted_output += "1. [Internal Knowledge Base (Search Fallback)](#)\n"
                return formatted_output
            except Exception as groq_err:
                logger.error(f"Groq fallback generation failed: {groq_err}")
                return "I tried to retrieve information for your query, but the generation service is temporarily unavailable."
                
        # 8. Synthesize answer from Tavily search results (Standard RAG path)
        if not self.groq_client:
            logger.warning("Groq client offline. Returning direct answer from Tavily search engine.")
            if tavily_direct_answer:
                llm_answer = tavily_direct_answer
            else:
                llm_answer = " ".join([r["body"] for r in results[:2]])
        else:
            # Format retrieved search results as LLM context
            context = ""
            for idx, r in enumerate(results):
                title = r.get("title", "No Title")
                body = r.get("body", "")
                context += f"Source [{idx + 1}]:\nTitle: {title}\nContent: {body}\n\n"
                
            prompt = f"""You are a precise and helpful AI assistant.
Your task is to answer the user's question using ONLY the provided web search results as context.

User Question: "{query}"

Web Search Results:
{context}

Instructions:
1. Provide a concise, direct, and factual answer to the question (usually 1-3 sentences).
2. Start your answer immediately with the information - do NOT use phrases like "Based on the search results...", "According to...", "The search results indicate...", etc.
3. If multiple sources agree on the answer, provide the answer confidently without mentioning agreement.
4. If the search results contain conflicting information, clearly state that conflicting details were found and describe the conflicting viewpoints.
5. If the search results do not contain enough information to answer the question, state clearly that you cannot find sufficient information.
6. Do not include raw source URLs, HTML tags, or markdown headers in your answer.
7. The answer must be completely grounded in the search results provided. Do not assume or extrapolate beyond the provided text.

Concise Direct Answer:"""

            logger.info("Sending search context to Groq for direct answer generation...")
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise answer generator that synthesizes search results into a single direct answer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=256
                )
                
                llm_answer = response.choices[0].message.content.strip()
                llm_answer = llm_answer.replace("---", "").replace("***", "").strip()
            except Exception as groq_err:
                logger.error(f"Groq generation with context failed: {groq_err}. Using Tavily direct answer fallback.")
                if tavily_direct_answer:
                    llm_answer = tavily_direct_answer
                else:
                    llm_answer = " ".join([r["body"] for r in results[:2]])
                    
        # Format final response: AI answer first, followed by sources at the bottom
        formatted_output = f"{llm_answer}\n\n"
        formatted_output += "---\n"
        formatted_output += "**Sources:**\n"
        for idx, r in enumerate(results):
            title = r.get("title", "No Title")
            href = r.get("href", "#")
            formatted_output += f"{idx + 1}. [{title}]({href})\n"
            
        # If debug mode is enabled, display raw search results at the end
        if getattr(settings, 'debug', False):
            formatted_output += "\n---\n### 🔍 Web Search Debug (Raw Results)\n"
            for idx, r in enumerate(results):
                title = r.get("title", "No Title")
                href = r.get("href", "#")
                body = r.get("body", "")
                formatted_output += f"{idx + 1}. **[{title}]({href})**\n"
                formatted_output += f"   {body}\n\n"
                
        return formatted_output
