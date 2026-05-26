from app.agents.state import AgentState


def fallback_node(
    state: AgentState
) -> AgentState:

    state["fallback"] = True

    state["response"] = (
        "Unable to process request."
    )

    return state