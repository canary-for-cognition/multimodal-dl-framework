import torch
from torch import nn

from classifier.classes.modules.sequences.rnn.RNN import RNN


class CNNRNN(nn.Module):

    def __init__(self, network_params: dict, activate: bool = True):
        super().__init__()

        rnn_params = network_params["rnn"]
        bidirectional = rnn_params["bidirectional"]
        output_size = rnn_params["output_size"]
        self._hidden_size = rnn_params["hidden_size"]
        rnn_params["input_size"] = self._hidden_size

        self._pre_activation_size = 2 * self._hidden_size if bidirectional else self._hidden_size
        self.__activate = activate

        self.cnn = None
        self.rnn = RNN(rnn_params, activation=False)

        if self.__activate:
            self.classifier = nn.Sequential(
                nn.Linear(self._pre_activation_size, self._hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self._hidden_size, output_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.rnn(x.unsqueeze(1))
        return self.classifier(x) if self.__activate else x

    def init_state(self, batch_size: int) -> tuple:
        return self.rnn.init_state(batch_size)

    def get_hidden_size(self) -> int:
        return self._hidden_size

    def get_pre_activation_size(self) -> int:
        return self._pre_activation_size
