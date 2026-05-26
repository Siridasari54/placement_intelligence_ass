from rank_bm25 import BM25Okapi


class BM25RetrieverCustom:

    def retrieve(self, docs, query):
        corpus = [doc.page_content for doc in docs]
        tokenized = [doc.split() for doc in corpus]

        bm25 = BM25Okapi(tokenized)

        scores = bm25.get_scores(query.split())

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for _, doc in ranked[:5]]