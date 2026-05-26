from app.agents.state import AgentState


def router_node(
    state: AgentState
) -> AgentState:

    prompt = state.get(
        "prompt",
        ""
    ).lower()

    if len(prompt.strip()) == 0:

        state["route"] = "fallback"

    else:

        state["route"] = "rewrite"

    return state