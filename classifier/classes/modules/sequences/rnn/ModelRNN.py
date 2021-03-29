from typing import Dict

import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.sequences.rnn.RNN import RNN


class ModelRNN(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        network_params["input_size"] = network_params["modality"]["num_features"]
        self._network = RNN(network_params).float().to(self._device)

    def predict(self, x: torch.Tensor, **kwargs: any) -> torch.Tensor:
        """
        Performs a prediction using the RNN and returns the output logits
        :param x: a batch of input sequences
        :return: the output logits
        """
        s = self._network.init_state(x.shape[0])
        return self._network(x, s)
