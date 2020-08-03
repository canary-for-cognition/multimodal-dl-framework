import os

import torch
import torch.nn as nn

from classifier.classes.modules.sequences.rnn.RNN import RNN
from classifier.classes.utils.AssetLoader import AssetLoader


class WordAttentionRNN(RNN):

    def __init__(self, network_params: dict, activation: bool = False):
        embedding_size = network_params["modality"]["embedding_size"]
        use_pre_trained_embeddings = network_params["use_pre_trained_embeddings"]
        network_params["input_size"] = embedding_size
        super().__init__(network_params, activation)

        if use_pre_trained_embeddings:
            base_path_to_embeddings = network_params["modality"]["path_to_pre_trained_embeddings"]
            embeddings_name = network_params["pre_trained_embeddings"]
            path_to_embeddings = os.path.join(base_path_to_embeddings, embeddings_name)
            embedding_weights = AssetLoader.load_embedding_weights(path_to_embeddings)
            self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        else:
            vocabulary_size = network_params["modality"]["vocabulary_size"]
            self.embedding = nn.Embedding(vocabulary_size + 1, embedding_size, padding_idx=0)
            self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, self._state = self._rnn(self.embedding(inputs), self._state)
        return self._attention(outputs)
