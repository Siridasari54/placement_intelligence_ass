class ContextRefiner:

    def refine(self, docs):
        refined = []

        for doc in docs:
            if len(doc.page_content.strip()) > 50:
                refined.append(doc)

        return refined