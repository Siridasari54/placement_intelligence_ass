import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import RAGPipeline
from core.di.factories import register_services
from core.di.container import ServiceContainer
from ui.query_handler import QueryHandler

def test():
    qh = QueryHandler()
    query = "who win the latest ipl"
    print(f"Executing Query: {query}")
    res = qh.execute_rag_query(query)
    print("\n--- RESULTS ---")
    print(res)

if __name__ == "__main__":
    test()
