class ContextFilter:

    def filter(self, docs):

        filtered = []

        for doc in docs:
            if len(doc.page_content.split()) > 10:
                filtered.append(doc)

        return filtered