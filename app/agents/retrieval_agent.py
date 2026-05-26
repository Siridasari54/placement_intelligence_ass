from app.agents.state import AgentState

from app.retrieval.hybrid_retriever import HybridRetriever

from app.vectorstore.vectorstore_manager import VectorStoreManager


def retrieval_node(
    state: AgentState
) -> AgentState:

    query = state.get(
        "rewritten_prompt",
        state["prompt"]
    )

    # Load vector store (if needed for other processes)
    vectorstore = VectorStoreManager().load()

    # Initialize HybridRetriever (BM25 retriever is internally created if not provided)
    retriever = HybridRetriever()

    # Perform retrieval; HybridRetriever internally uses the vector store manager
    retrieved_docs = retriever.retrieve(
        query=query,
        k=5,
        filter=None
    )

    state["retrieved_docs"] = retrieved_docs

    return state