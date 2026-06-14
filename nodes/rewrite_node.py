"""Query rewrite node for LangGraph."""

from graph.state import PlacementState
from groq import Groq
from config.settings import settings


def rewrite_node(state: PlacementState) -> PlacementState:
    """Rewrite query for better retrieval when confidence is low.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with rewritten query and incremented retry count
    """
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    
    # Prevent infinite loops - only allow one rewrite attempt
    if retry_count >= 1:
        state["fallback"] = True
        return state
    
    try:
        client = Groq(api_key=settings.groq_api_key)
        
        prompt = f"""Rewrite the following query to be clearer and more focused on placement data.
Keep the meaning the same but make it more specific for a placement intelligence system.

Original query: "{query}"

Rewritten query:"""
        
        response = client.chat.completions.create(
            model=settings.generation.model,
            messages=[
                {"role": "system", "content": "You are a query rewriter for placement data. Output only the rewritten query, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        rewritten_query = response.choices[0].message.content.strip()
        
        # Update state
        state["query"] = rewritten_query
        state["retry_count"] = retry_count + 1
        state["fallback"] = False  # Reset fallback flag for retry
        
        return state
        
    except Exception as e:
        # If rewrite fails, proceed with original query
        state["retry_count"] = retry_count + 1
        state["fallback"] = True
        return state
