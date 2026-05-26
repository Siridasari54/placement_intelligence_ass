class HallucinationFilter:

    def filter(self, response, docs):

        if len(docs) == 0:
            return "I could not find relevant information."

        return response