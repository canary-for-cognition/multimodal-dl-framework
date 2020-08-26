from typing import Union

import torch
from torch import nn

from classifier.classes.modules.base.AttentionNN import AttentionNN


class RNN(AttentionNN):

    def __init__(self, network_params: dict, activation: bool = True):

        self.__rnn_type = network_params["type"]
        self.__output_size = network_params["output_size"] if "output_size" in network_params.keys() else None
        self._use_attention = network_params["attention"] if "attention" in network_params.keys() else False
        self._stateless = network_params["stateless"] if "stateless" in network_params.keys() else True
        self._use_encoding = network_params["encoding"]["active"] if "encoding" in network_params.keys() else False
        self._normalized = network_params["normalized"] if "normalized" in network_params.keys() else False
        self._input_size = network_params["input_size"]
        self._hidden_size = network_params["hidden_size"]
        self._num_layers = network_params["num_layers"]
        self.__dropout = network_params["dropout"]
        self._bidirectional = network_params["bidirectional"]
        self._batch_first = network_params["batch_first"]
        self._activation = activation

        self.__linear_size = 2 * self._hidden_size if self._bidirectional else self._hidden_size
        super().__init__(self.__linear_size)

        self._rnn = self.__select_rnn()
        self._state = None

        if self._normalized:
            self._normalization_layer = nn.Sequential(
                nn.LayerNorm(self._input_size),
                nn.GELU(),
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
                                               batch_first=self._batch_first)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: the input sequences of shape [batch_size, sequence_length, num_features]
        :return: the logit for prediction with respect to the input data
        """
        inputs = self._normalization_layer(inputs) if self._normalized else inputs
        inputs = self._encoding(inputs) if self._use_encoding else inputs

        self._state = None if self._stateless else self._state
        outputs, self._state = self._rnn(inputs.permute(1, 0, 2), self._state)

        outputs = self._attention(outputs) if self._use_attention else outputs

        outputs = outputs[-1, :, :]
        return self._fc(outputs) if self._activation else outputs

    def set_state(self, state: Union[tuple, torch.Tensor]):
        self._state = state

    def get_state(self) -> Union[tuple, torch.Tensor]:
        return self._state

    def get_hidden_size(self) -> int:
        return 2 * self._hidden_size if self._bidirectional else self._hidden_size

    def init_state(self, batch_size: int) -> Union[tuple, torch.Tensor]:
        initialization = torch.zeros(self.__linear_size, batch_size, self._hidden_size, requires_grad=True)
        return initialization if self.__rnn_type != "lstm" else initialization, initialization
