import torch
import torch.nn as nn

from classifier.classes.modules.text.han.submodules.SentenceAttentionRNN import SentenceAttentionRNN
from classifier.classes.modules.text.han.submodules.WordAttentionRNN import WordAttentionRNN


class HAN(nn.Module):

    def __init__(self, network_params: dict, activation: bool = True):
        super().__init__()

        modality_params = network_params["modality"]

        word_attention_params = network_params["word_attention_rnn"]
        word_attention_params["modality"] = modality_params

        sentence_attention_params = network_params["sentence_attention_rnn"]
        sentence_attention_params["word_hidden_size"] = word_attention_params["hidden_size"]

        self.__max_sentences = modality_params["max_sentences"]

        self.word_attention_rnn = WordAttentionRNN(word_attention_params)
        self.sentence_attention_rnn = SentenceAttentionRNN(sentence_attention_params, activation)

    def __compute_word_attention(self, text: torch.Tensor) -> torch.Tensor:
        attentions = None
        for i in range(self.__max_sentences):
            attention = self.word_attention_rnn(text[i, :, :].transpose(0, 1))
            attentions = attention if attentions is None else torch.cat((attentions, attention), 0)
        return attentions

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.sentence_attention_rnn(self.__compute_word_attention(text.permute(1, 0, 2)))

    def init_state(self, batch_size: int) -> tuple:
        return self.word_attention_rnn.init_state(batch_size), self.sentence_attention_rnn.init_state(batch_size)

    def get_hidden_size(self) -> int:
        return self.sentence_attention_rnn.get_hidden_size()
