from typing import Union

import torch
from torch import nn

from classifier.classes.modules.base.AttentionNN import AttentionNN


class RNN(AttentionNN):

    def __init__(self, network_params: dict, activation: bool = True):

        self.__device = network_params["device"]
        self.__rnn_type = network_params["type"]
        self.__dropout = network_params["dropout"]
        self.__output_size = network_params["output_size"] if "output_size" in network_params.keys() else None
        self._use_attention = network_params["attention"] if "attention" in network_params.keys() else False
        self._use_encoding = network_params["encoding"]["active"] if "encoding" in network_params.keys() else False
        self._normalized = network_params["normalized"] if "normalized" in network_params.keys() else False
        self._input_size = network_params["input_size"]
        self._hidden_size = network_params["hidden_size"]
        self._num_layers = network_params["num_layers"]
        self._bidirectional = network_params["bidirectional"]

        self._activation = activation

        self.__linear_size = 2 * self._hidden_size if self._bidirectional else self._hidden_size

        super().__init__(self.__linear_size)

        self._rnn = self.__select_rnn()

        if self._normalized:
            self._normalization_layer = nn.Sequential(
                nn.LayerNorm(self._input_size),
                nn.GELU()
            )

        if self._use_encoding:
            encoding_output_size = network_params["encoding"]["output_size"]
            self._encoding = nn.Linear(self._input_size, encoding_output_size)
            self._input_size = encoding_output_size

        if self._activation:
            self._fc = nn.Linear(self.__linear_size, self.__output_size)

    def __select_rnn(self) -> nn.Module:
        rnn_types_maps = {
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }
        return rnn_types_maps[self.__rnn_type](input_size=self._input_size,
                                               hidden_size=self._hidden_size,
                                               num_layers=self._num_layers,
                                               bidirectional=self._bidirectional,
                                               dropout=self.__dropout,
                                               batch_first=True)

    def get_hidden_size(self) -> int:
        return 2 * self._hidden_size if self._bidirectional else self._hidden_size

    def init_state(self, batch_size: int) -> Union[tuple, torch.Tensor]:
        num_layers = 2 * self._num_layers if self._bidirectional else self._num_layers
        s = torch.zeros(num_layers, batch_size, self._hidden_size)
        return s.to(self.__device) if self.__rnn_type != "lstm" else (s.to(self.__device), s.to(self.__device))

    def forward(self, x: torch.Tensor, s: Union[torch.Tensor, tuple]) -> torch.Tensor:
        """
        :param x: the input sequences of shape [batch_size, sequence_length, num_features]
        :param s: the hidden state of the RNN [num_layers, batch_size, hidden_size]
        :return: the logit for prediction with respect to the input data
        """
        x = self._normalization_layer(x) if self._normalized else x
        x = self._encoding(x) if self._use_encoding else x

        o, _ = self._rnn(x, s)

        o = self._attention(o) if self._use_attention else o

        o = o[:, -1, :]
        return self._fc(o) if self._activation else o
