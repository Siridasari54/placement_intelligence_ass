"""Multi-hop retrieval node for comparison queries."""

from graph.state import PlacementState
from core.di.container import ServiceContainer
from core.di.factories import register_services
from retrieval.multi_hop_retriever import MultiHopRetriever


def multi_hop_node(state: PlacementState) -> PlacementState:
    """Perform multi-hop retrieval for comparison queries.
    
    Retrieves documents for each entity in comparison queries and merges them.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with merged documents from multiple retrieval hops
    """
    if not state.get("is_comparison", False):
        # Not a comparison query, proceed normally
        return state
    
    entities = state.get("comparison_entities", [])
    if len(entities) < 2:
        # Need at least 2 entities for comparison
        return state
    
    try:
        # Initialize container and retriever
        container = ServiceContainer()
        register_services(container)
        retriever = container.get_service(MultiHopRetriever)
        
        if not retriever:
            return state
        
        query = state["query"]
        all_documents = []
        
        # Retrieve for each entity (max 2 hops)
        for entity in entities[:2]:
            entity_query = f"{query} {entity}"
            docs = retriever.retrieve(entity_query, top_k=5)
            all_documents.extend(docs)
        
        # Merge documents and update state
        state["documents"] = all_documents
        
        return state
        
    except Exception as e:
        # If multi-hop fails, proceed with normal flow
        return state
