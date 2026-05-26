from app.agents.state import AgentState

from app.pipeline.generate.citation_builder import (
    CitationBuilder
)


def citation_node(
    state: AgentState
) -> AgentState:

    docs = state.get(
        "refined_docs",
        []
    )

    citations = CitationBuilder().build(
        docs
    )

    state["citations"] = citations

    return state