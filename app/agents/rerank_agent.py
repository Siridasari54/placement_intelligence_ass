from app.agents.state import AgentState

from app.pipeline.rerank.reranker import (
    Reranker
)


def rerank_node(
    state: AgentState
) -> AgentState:

    docs = state.get(
        "retrieved_docs",
        []
    )

    reranker = Reranker()

    reranked_docs = reranker.rerank(
        docs
    )

    state["reranked_docs"] = reranked_docs

    return state