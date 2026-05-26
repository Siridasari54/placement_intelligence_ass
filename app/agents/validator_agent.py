from app.agents.state import AgentState


def validator_node(
    state: AgentState
) -> AgentState:

    response = state.get(
        "response",
        ""
    )

    docs = state.get(
        "refined_docs",
        []
    )

    if len(response.strip()) == 0:

        state["validation"] = False

    elif len(docs) == 0:

        state["validation"] = False

    else:

        state["validation"] = True

    return state