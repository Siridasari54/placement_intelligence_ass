from safety.fallback_guard import FallbackGuard

guard = FallbackGuard()

def confidence_node(state):
    confidence = guard.check_confidence(
        state["query"],
        state["documents"]
    )

    return {
        "confidence": confidence,
        "fallback": guard.should_fallback(confidence)
    }