from app.agents.state import AgentState

from app.pipeline.refine.context_refiner import (
    ContextRefiner
)


def refine_node(
    state: AgentState
) -> AgentState:

    docs = state.get(
        "reranked_docs",
        []
    )

    refined_docs = (
        ContextRefiner().refine(docs)
    )

    state["refined_docs"] = refined_docs

    return state