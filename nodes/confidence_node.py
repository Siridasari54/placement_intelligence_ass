"""Confidence node for LangGraph - enhanced with retry logic."""

from graph.state import PlacementState
from safety.fallback_guard import FallbackGuard

guard = FallbackGuard()


def confidence_node(state: PlacementState) -> PlacementState:
    """Check confidence and determine if fallback or rewrite is needed.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with confidence score and fallback decision
    """
    confidence = guard.check_confidence(
        state["query"],
        state["documents"]
    )

    state["confidence"] = confidence
    state["fallback"] = guard.should_fallback(confidence)
    
    # Ensure retry_count is initialized
    if "retry_count" not in state:
        state["retry_count"] = 0
    
    return state