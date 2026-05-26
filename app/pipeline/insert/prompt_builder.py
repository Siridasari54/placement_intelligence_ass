class PromptBuilder:

    def build(self, query, docs):

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are an intelligent AI assistant.

Answer the question only using the context below.

Question:
{query}

Context:
{context}

Provide accurate answer with citations.
"""

        return prompt