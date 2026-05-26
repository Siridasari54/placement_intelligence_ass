from app.agents.state import AgentState


def evaluation_node(
    state: AgentState
) -> AgentState:

    evaluation = {

        "response_length": len(
            state.get("response", "")
        ),

        "retrieved_docs": len(
            state.get("retrieved_docs", [])
        ),

        "validation": state.get(
            "validation",
            False
        )
    }

    state["evaluation"] = evaluation

    return state