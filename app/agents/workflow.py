from langgraph.graph import (
    StateGraph,
    START,
    END
)

from app.agents.state import AgentState

from app.agents.router_agent import router_node
from app.agents.rewrite_agent import rewrite_node
from app.agents.retrieval_agent import retrieval_node
from app.agents.rerank_agent import rerank_node
from app.agents.refine_agent import refine_node
from app.agents.insert_agent import insert_node
from app.agents.summarizer_agent import summarizer_node
from app.agents.citation_agent import citation_node
from app.agents.validator_agent import validator_node
from app.agents.fallback_agent import fallback_node
from app.agents.evaluation_agent import evaluation_node


workflow = StateGraph(
    AgentState
)


# ADD NODES

workflow.add_node(
    "router",
    router_node
)

workflow.add_node(
    "rewrite",
    rewrite_node
)

workflow.add_node(
    "retriever",
    retrieval_node
)

workflow.add_node(
    "reranker",
    rerank_node
)

workflow.add_node(
    "refine",
    refine_node
)

workflow.add_node(
    "insert",
    insert_node
)

workflow.add_node(
    "summarizer",
    summarizer_node
)

workflow.add_node(
    "citation",
    citation_node
)

workflow.add_node(
    "validator",
    validator_node
)

workflow.add_node(
    "fallback",
    fallback_node
)

workflow.add_node(
    "evaluator",
    evaluation_node
)


# START

workflow.add_edge(
    START,
    "router"
)


# ROUTING

def route_decision(
    state: AgentState
):

    route = state.get(
        "route",
        "rewrite"
    )

    if route == "fallback":
        return "fallback"

    return "rewrite"


workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "fallback": "fallback",
        "rewrite": "rewrite"
    }
)


# PIPELINE

workflow.add_edge(
    "rewrite",
    "retriever"
)

workflow.add_edge(
    "retriever",
    "reranker"
)

workflow.add_edge(
    "reranker",
    "refine"
)

workflow.add_edge(
    "refine",
    "insert"
)

workflow.add_edge(
    "insert",
    "summarizer"
)

workflow.add_edge(
    "summarizer",
    "citation"
)

workflow.add_edge(
    "citation",
    "validator"
)

workflow.add_edge(
    "validator",
    "evaluator"
)

workflow.add_edge(
    "fallback",
    "evaluator"
)

workflow.add_edge(
    "evaluator",
    END
)


graph_app = workflow.compile()