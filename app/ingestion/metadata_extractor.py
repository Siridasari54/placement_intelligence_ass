class MetadataExtractor:

    def extract(self, docs, source_name):
        for doc in docs:
            doc.metadata["source"] = source_name

        return docs