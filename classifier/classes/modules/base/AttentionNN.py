import torch
import torch.nn.functional as F
from torch import nn


class AttentionNN(nn.Module):

    def __init__(self, linear_size: int):
        super().__init__()

        self.__bias = nn.Parameter(torch.rand(linear_size, 1), requires_grad=True)

        self.__weights = nn.Parameter(torch.rand(linear_size, linear_size), requires_grad=True)
        self.__weights.data.uniform_(-0.1, 0.1)

        self.__weights_projection = nn.Parameter(torch.rand(linear_size, 1), requires_grad=True)
        self.__weights_projection.data.uniform_(-0.1, 0.1)

    def _attention(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention matrix for the RNN outputs
        :param outputs: the outputs of the RNN
        :return: the outputs of the RNN weighted by the attention matrix
        """
        batch_attention = self.__batch_mm(self.__batch_mm_bias(outputs))
        attention_weights = F.softmax(batch_attention.transpose(1, 0), dim=1).transpose(1, 0)
        return self.__attention_mm(outputs, attention_weights)

    def __batch_mm_bias(self, sequence: torch.Tensor) -> torch.Tensor:
        product = None
        for i in range(sequence.shape[0]):
            s_product = torch.mm(sequence[i], self.__weights)
            s_bias = self.__bias.expand(self.__bias.shape[0], s_product.shape[0]).transpose(0, 1)
            s_product_bias = torch.tanh(s_product + s_bias).unsqueeze(0)
            product = s_product_bias if product is None else torch.cat((product, s_product_bias), 0)
        return product.squeeze()

    def __batch_mm(self, sequence: torch.Tensor) -> torch.Tensor:
        product = None
        for i in range(sequence.size(0)):
            item = sequence[i]
            s_product = torch.mm(item.unsqueeze(0) if len(item.shape) == 1 else item, self.__weights_projection)
            s_product = s_product.unsqueeze(0)
            product = s_product if product is None else torch.cat((product, s_product), 0)
        return product.squeeze(1) if len(sequence[0].shape) == 1 else product.squeeze()

    @staticmethod
    def __attention_mm(rnn_outputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        attentions = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = weights[i].unsqueeze(1).expand_as(h_i)
            h_i = (a_i * h_i).unsqueeze(0)
            attentions = h_i if attentions is None else torch.cat((attentions, h_i), 0)
        return torch.sum(attentions, 0).unsqueeze(0)
