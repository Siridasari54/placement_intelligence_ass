class Deduplicator:

    def deduplicate(self, docs):
        seen = set()
        unique_docs = []

        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs