from app.pipeline.retrieve.vector_retriever import VectorRetriever
from app.pipeline.retrieve.bm25_retriever import BM25RetrieverCustom


class HybridRetriever:

    def retrieve(self, vectorstore, docs, query):
        vector_docs = VectorRetriever().retrieve(vectorstore, query)
        bm25_docs = BM25RetrieverCustom().retrieve(docs, query)

        combined = vector_docs + bm25_docs

        unique = []
        seen = set()

        for doc in combined:
            if doc.page_content not in seen:
                unique.append(doc)
                seen.add(doc.page_content)

        return unique