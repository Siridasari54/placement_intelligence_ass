"""Router node for LangGraph - enhanced with comparison detection."""

from graph.state import PlacementState


def router_node(state: PlacementState) -> PlacementState:
    """Route query to appropriate tool and detect comparison queries.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with route and comparison detection
    """
    query = state["query"].lower()
    
    # Initialize new state fields
    state["original_query"] = state.get("original_query", state["query"])
    state["retry_count"] = state.get("retry_count", 0)
    state["is_comparison"] = False
    state["comparison_entities"] = []
    
    # Detect comparison queries
    comparison_keywords = ["compare", "versus", "vs", "difference", "better", "between"]
    if any(keyword in query for keyword in comparison_keywords):
        state["is_comparison"] = True
        # Simple entity extraction - look for company names
        companies = ["tcs", "infosys", "wipro", "google", "amazon", "microsoft", "meta", "apple"]
        found_entities = [company for company in companies if company in query]
        state["comparison_entities"] = found_entities[:2]  # Limit to 2 entities
    
    # Tool routing logic
    if any(word in query for word in [
        "cgpa",
        "student",
        "package",
        "backlog",
        "analytics",
        "summary"
    ]):
        state["route"] = "database"
        return state

    if any(word in query for word in [
        "latest",
        "today",
        "current",
        "ipl",
        "stock",
        "population",
        "weather",
        "temperature"
    ]):
        state["route"] = "web"
        return state

    state["route"] = "rag"
    return state