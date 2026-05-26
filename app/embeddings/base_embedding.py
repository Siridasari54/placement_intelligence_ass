from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @abstractmethod
    def load(self):
        pass