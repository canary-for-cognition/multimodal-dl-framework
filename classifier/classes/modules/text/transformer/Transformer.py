import os

import torch
from torch import nn
from transformers import BertModel, RobertaModel


class Transformer(nn.Module):

    def __init__(self, network_params: dict, activation: bool = True):
        super().__init__()

        pre_trained_architecture = network_params["pre_trained_architecture"]
        self.__pre_trained_model_type = network_params["pre_trained_model"]
        self.__dropout = network_params["dropout"]
        self.__output_size = network_params["output_size"]
        self.__activation = activation

        use_local_pre_trained_model = network_params["load_local_pre_trained_model"]
        if use_local_pre_trained_model:
            path_to_pre_trained_models = network_params["modality"]["path_to_pre_trained_models"]
            self.__pre_trained_model_type = os.path.join(path_to_pre_trained_models, self.__pre_trained_model_type)

        model = self.__select_pre_trained_architecture(pre_trained_architecture)
        self._transformer = model.from_pretrained(self.__pre_trained_model_type)
        self._hidden_size = self._transformer.config.hidden_size

        if self.__activation:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.__dropout),
                nn.Linear(self._hidden_size, self.__output_size),
                nn.Softmax(dim=1)
            )

    def __select_pre_trained_architecture(self, transformer_type: str) -> nn.Module:
        transformers_map = {
            "bert": BertModel,
            "roberta": RobertaModel
        }
        return transformers_map[transformer_type].from_pretrained(self.__pre_trained_model_type)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, outputs = self._transformer(input_ids, attention_mask)
        return self.classifier(outputs) if self.__activation else outputs

    def get_hidden_size(self) -> int:
        return self._hidden_size
