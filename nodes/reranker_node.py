from core.interfaces import IReranker
from core.di.factories import register_services
from core.di.container import ServiceContainer

# Initialize reranker through DI container
container = ServiceContainer()
register_services(container)
reranker = container.get_service(IReranker)

def reranker_node(state):
    docs = reranker.rerank(
        state["query"],
        state["documents"],
        top_k=5
    )

    return {"documents": docs}