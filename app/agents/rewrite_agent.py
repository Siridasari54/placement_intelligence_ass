from app.agents.state import AgentState

from app.pipeline.rewrite.query_rewriter import (
    QueryRewriter
)


def rewrite_node(
    state: AgentState
) -> AgentState:

    query = state["prompt"]

    rewriter = QueryRewriter()

    rewritten_query = rewriter.rewrite(
        query
    )

    state["rewritten_prompt"] = rewritten_query

    return state