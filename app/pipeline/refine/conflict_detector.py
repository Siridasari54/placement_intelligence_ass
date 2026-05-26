class ConflictDetector:

    def detect(self, docs):

        conflicts = []

        seen = {}

        for doc in docs:
            text = doc.page_content[:100]

            if text in seen:
                conflicts.append(doc)

            seen[text] = True

        return conflicts