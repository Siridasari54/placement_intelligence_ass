class ResponseCache:

    def __init__(self):

        self.responses = {}

    def save_response(
        self,
        query,
        response
    ):

        self.responses[query] = response

    def get_response(
        self,
        query
    ):

        return self.responses.get(query)