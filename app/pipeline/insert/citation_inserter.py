class CitationInserter:

    def insert(self, response, citations):

        citation_text = "".join(citations)

        return f"{response}{citation_text}"