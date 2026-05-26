class CitationBuilder:

    def build(self, docs):

        citations = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                0
            )

            citation = (
                f"Source: {source} | Page: {page}"
            )

            citations.append(citation)

        return citations