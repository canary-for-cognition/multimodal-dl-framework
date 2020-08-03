import torch
from gensim.models import KeyedVectors


class AssetLoader:

    @staticmethod
    def load_embedding_weights(path_to_pre_trained_embeddings: str) -> torch.Tensor:
        embedding_weights = KeyedVectors.load_word2vec_format(path_to_pre_trained_embeddings)
        return torch.FloatTensor(embedding_weights.vectors)
