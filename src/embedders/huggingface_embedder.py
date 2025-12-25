
from src.embedders.base_embedder import BaseEmbedder
from sentence_transformers import SentenceTransformer


class HuggingfaceEmbedder(BaseEmbedder):

    def __init__(self, model_name):
        super().__init__()
        self.model = SentenceTransformer(model_name)


    def embed_text(self, text:str) -> list[int]:
        if text is not isinstance(str):
            raise TypeError("Text must be a string ")
        return self.model.encode(text)
    
