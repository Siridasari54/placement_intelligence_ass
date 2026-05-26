import sys
sys.path.append(r'c:/Users/sirid/OneDrive/Desktop/placement_rag_sys')

from app.services.ingestion_service import ingestion_service
from app.retrieval.hybrid_retriever import HybridRetriever

# Assume ingestion already performed; use existing BM25 retriever set globally
retriever = HybridRetriever()
results = retriever.retrieve(query='test query', k=3)
print('Retrieved', len(results), 'results')
for i, doc in enumerate(results, 1):
    print(f"Result {i}:", doc.get('content', doc))
