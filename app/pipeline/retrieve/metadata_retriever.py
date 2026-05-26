class MetadataRetriever:

    def filter(self, docs, source=None):

        if source is None:
            return docs

        filtered = []

        for doc in docs:
            if doc.metadata.get("source") == source:
                filtered.append(doc)

        return filtered