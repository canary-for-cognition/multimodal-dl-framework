import torch

from classifier.classes.modules.sequences.rnn.RNN import RNN


class SentenceAttentionRNN(RNN):

    def __init__(self, network_params: dict, activation: bool = True):
        word_hidden_size = network_params["word_hidden_size"]
        bidirectional = network_params["bidirectional"]
        network_params["input_size"] = 2 * word_hidden_size if bidirectional else word_hidden_size
        super().__init__(network_params, activation)

    def forward(self, word_attentions: torch.Tensor) -> torch.Tensor:
        outputs, self._state = self._rnn(word_attentions, self._state)
        attention_weighted_outputs = self._attention(outputs).squeeze(0)
        return self._fc(attention_weighted_outputs) if self._activation else attention_weighted_outputs
