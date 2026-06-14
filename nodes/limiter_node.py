from safety.overshadow_limiter import OvershadowLimiter

limiter = OvershadowLimiter()

def limiter_node(state):
    docs, overshadow_risk = limiter.limit_context(
        state["documents"]
    )

    return {"documents": docs}