import os
from typing import Dict

import torch
from torch import nn
from transformers import BertModel, RobertaModel


class Transformer(nn.Module):

    def __init__(self, network_params: Dict, activation: bool = True):
        super().__init__()


        model_type = network_params["model_type"]
        pretrained_model_type = network_params["pretrained_model"]
        reload_pretrained = network_params["reload_pretrained"]
        output_size = network_params["output_size"]
        dropout = network_params["dropout"]
        self.__activation = activation

        if reload_pretrained:
            path_to_pretrained_models = network_params["modality"]["path_to_pretrained_models"]
            pretrained_model_type = os.path.join(path_to_pretrained_models, pretrained_model_type)

        transformers_map = {"bert": BertModel, "roberta": RobertaModel}
        self._transformer = transformers_map[model_type].from_pretrained(pretrained_model_type)
        self.__hidden_size = self._transformer.config.hidden_size

        if self.__activation:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.__hidden_size, output_size),
                nn.Softmax(dim=1)
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, o = self._transformer(input_ids, attention_mask)
        return self.classifier(o) if self.__activation else o

    def get_hidden_size(self) -> int:
        return self.__hidden_size
