class ScoreFusion:

    def fuse(self, vector_docs, bm25_docs):

        combined = vector_docs + bm25_docs

        unique = []
        seen = set()

        for doc in combined:
            if doc.page_content not in seen:
                unique.append(doc)
                seen.add(doc.page_content)

        return unique