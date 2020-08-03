from typing import Union

import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.sequences.rnn.RNN import RNN


class ModelRNN(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        network_params["input_size"] = network_params["modality"]["num_features"]
        self._send_network_to_device(RNN(network_params))

    def predict(self,
                inputs: torch.Tensor,
                state: Union[tuple, torch.Tensor] = None,
                **kwargs: any) -> Union[tuple, torch.Tensor]:
        """
        Performs a prediction using the RNN and returns the output logits
        :param inputs: a batch of input sequences
        :param state: an optional state (for TBPTT)
        :return: the output logits and the state for TBPTT (if a state was provided)
        """
        self._network.set_state(state)
        outputs = self._network(inputs.float().to(self._device))
        return outputs if state is None else (outputs, self._network.get_state())
