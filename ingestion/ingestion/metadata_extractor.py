class MetadataExtractor:

    def extract(self, docs, source_name):
        for doc in docs:
            # Handle both Document objects and dictionary formats
            if hasattr(doc, 'metadata'):
                doc.metadata["source"] = source_name
            elif isinstance(doc, dict):
                doc["metadata"] = doc.get("metadata", {})
                doc["metadata"]["source"] = source_name
            else:
                # If it's neither, skip or handle appropriately
                continue

        return docs