from langgraph.graph import StateGraph, END

from graph.state import PlacementState

from nodes.router_node import router_node
from nodes.retrieval_node import retrieval_node
from nodes.reranker_node import reranker_node
from nodes.conflict_node import conflict_node
from nodes.limiter_node import limiter_node
from nodes.confidence_node import confidence_node
from nodes.generation_node import generation_node
from nodes.fallback_node import fallback_node


builder = StateGraph(PlacementState)

builder.add_node("router", router_node)

builder.add_node("retrieve", retrieval_node)

builder.add_node("rerank", reranker_node)

builder.add_node("conflict", conflict_node)

builder.add_node("limit", limiter_node)

builder.add_node("confidence", confidence_node)

builder.add_node("generate", generation_node)

builder.add_node("fallback", fallback_node)


builder.set_entry_point("router")


builder.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {
        "rag": "retrieve",
        "database": "generate",
        "web": "generate",
    }
)

builder.add_edge("retrieve", "rerank")

builder.add_edge("rerank", "conflict")

builder.add_edge("conflict", "limit")

builder.add_edge("limit", "confidence")


builder.add_conditional_edges(
    "confidence",
    lambda s: s["fallback"],
    {
        True: "fallback",
        False: "generate",
    }
)

builder.add_edge("generate", END)

builder.add_edge("fallback", END)


graph = builder.compile()