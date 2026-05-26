from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, docs, top_k=5):

        if not docs:
            return []

        pairs = []

        query = docs[0].metadata.get("query", "")

        for doc in docs:
            pairs.append([query, doc.page_content])

        scores = self.model.predict(pairs)

        scored_docs = list(zip(scores, docs))

        scored_docs.sort(
            key=lambda x: x[0],
            reverse=True
        )

        reranked_docs = [
            doc for score, doc in scored_docs[:top_k]
        ]

        return reranked_docs