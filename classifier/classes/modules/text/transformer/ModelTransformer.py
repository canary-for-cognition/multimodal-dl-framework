import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.text.transformer.Transformer import Transformer


class ModelTransformer(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        self._network = Transformer(network_params).float().to(self._device)

    def predict(self, x: tuple, **kwargs) -> torch.Tensor:
        input_ids = x[0].to(self._device)
        attention_mask = x[1].to(self._device)
        return self._network(input_ids, attention_mask)
