class RetrievalEval:

    def evaluate(self, retrieved_docs):

        return {
            "retrieved_documents": len(retrieved_docs)
        }