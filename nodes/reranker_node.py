from retrieval.reranker import Reranker
from core.di.factories import register_services
from core.di.container import ServiceContainer

# Initialize reranker through DI container
container = ServiceContainer()
register_services(container)
reranker = container.get_service(Reranker)

def reranker_node(state):
    docs = reranker.rerank(
        state["query"],
        state["documents"],
        top_k=5
    )

    return {"documents": docs}