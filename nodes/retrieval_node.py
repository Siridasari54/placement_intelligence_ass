from retrieval.multi_hop_retriever import MultiHopRetriever
from core.di.factories import register_services
from core.di.container import ServiceContainer

# Initialize retriever through DI container
container = ServiceContainer()
register_services(container)
retriever = container.get_service(MultiHopRetriever)

def retrieval_node(state):
    docs = retriever.retrieve(
        state["query"],
        k=10
    )

    return {"documents": docs}