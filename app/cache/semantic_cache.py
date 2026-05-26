import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import (
    SentenceTransformer
)


class SemanticCache:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.cache = []

    def add(
        self,
        query,
        response
    ):

        embedding = self.model.encode(
            query
        )

        self.cache.append({

            "query": query,

            "response": response,

            "embedding": embedding
        })

    def search(
        self,
        query,
        threshold=0.90
    ):

        if len(self.cache) == 0:

            return None

        query_embedding = self.model.encode(
            query
        )

        best_score = 0

        best_response = None

        for item in self.cache:

            score = cosine_similarity(

                [query_embedding],

                [item["embedding"]]

            )[0][0]

            if score > best_score:

                best_score = score

                best_response = item["response"]

        if best_score >= threshold:

            return best_response

        return None