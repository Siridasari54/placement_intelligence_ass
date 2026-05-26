class FallbackHandler:

    def handle(self, response):

        if response is None:
            return "No response generated."

        if len(response.strip()) == 0:
            return "Empty response generated."

        return response