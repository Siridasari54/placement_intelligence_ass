"""Web Search tool for fetching live information outside the placement dataset."""

import logging
from typing import Dict, Any, List
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Fetches live information from the web via DuckDuckGo when queries are out-of-corpus."""
    
    def __init__(self):
        logger.info("WebSearchTool initialized")
        
    def execute(self, query: str) -> str:
        """Execute web search for the query and format results in Markdown.
        
        Args:
            query: User search query
            
        Returns:
            Formatted Markdown string of search results
        """
        logger.info(f"WebSearchTool executing query: {query}")
        
        try:
            # We use backend='html' as it is the most reliable backend in this environment
            with DDGS() as ddgs:
                results = list(ddgs.text(query, backend='html', max_results=3))
                
            if not results:
                return f"### 🌐 Web Search: {query}\nNo search results found on the web."
                
            md = f"### 🌐 Live Web Search: *\"{query}\"*\n\n"
            md += "Here are the latest findings from public web sources:\n\n"
            
            for idx, r in enumerate(results):
                title = r.get("title", "No Title")
                href = r.get("href", "#")
                body = r.get("body", "")
                md += f"{idx + 1}. **[{title}]({href})**\n"
                md += f"   {body}\n\n"
                
            md += "*Note: This information is fetched in real-time from the web and is not part of the static placement dataset.*"
            return md
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}", exc_info=True)
            return f"### 🌐 Web Search: {query}\nI tried to search the web for your query but encountered an issue: {str(e)}"
