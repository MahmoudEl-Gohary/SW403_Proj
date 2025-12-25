from abc import ABC, abstractmethod

class BaseEmbedder(ABC):

    @abstractmethod
    def embedText(self, text:str) -> None:
        """
        Creates embedding from texts

        Args:
            text (str): text to embed.
        """
