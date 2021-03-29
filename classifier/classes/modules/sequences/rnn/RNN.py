from typing import Union, Dict, Tuple

import torch
from torch import nn


class RNN(nn.Module):

    def __init__(self, network_params: Dict, activation: bool = True):
        super().__init__()

        self.__device = network_params["device"]
        self.__rnn_type = network_params["type"].upper()
        self.__num_layers = network_params["num_layers"]
        self.__bidirectional = network_params["bidirectional"]
        self.__hidden_size = network_params["hidden_size"]
        dropout = network_params["dropout"]
        input_size = network_params["input_size"]
        output_size = network_params["output_size"] if "output_size" in network_params.keys() else None

        self._activation = activation

        self._rnn = getattr(nn, self.__rnn_type)(input_size, self.__hidden_size, self.__num_layers,
                                                 batch_first=True, dropout=dropout, bidirectional=self.__bidirectional)

        if self._activation:
            linear_size = 2 * self.__hidden_size if self.__bidirectional else self.__hidden_size
            self._fc = nn.Linear(linear_size, output_size)

    def get_hidden_size(self) -> int:
        return 2 * self.__hidden_size if self.__bidirectional else self.__hidden_size

    def init_state(self, batch_size: int) -> Union[tuple, torch.Tensor]:
        num_layers = 2 * self.__num_layers if self.__bidirectional else self.__num_layers
        s = torch.zeros(num_layers, batch_size, self.__hidden_size)
        return s.to(self.__device) if self.__rnn_type != "LSTM" else (s.to(self.__device), s.to(self.__device))

    def forward(self, x: torch.Tensor, s: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """
        :param x: the input sequences of shape [batch_size, sequence_length, num_features]
        :param s: the hidden state of the RNN [num_layers, batch_size, hidden_size]
        :return: the logit for prediction with respect to the input data
        """
        o, _ = self._rnn(x, s)
        o = o[:, -1, :]
        return self._fc(o) if self._activation else o
