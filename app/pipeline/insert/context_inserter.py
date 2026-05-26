class ContextInserter:

    def insert(self, query, context):

        return f"""
Question:
{query}

Context:
{context}
"""