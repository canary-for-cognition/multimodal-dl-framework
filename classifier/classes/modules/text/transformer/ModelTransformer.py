import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.text.transformer.Transformer import Transformer


class ModelTransformer(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        self._send_network_to_device(Transformer(network_params))

    def predict(self, inputs: tuple, **kwargs) -> torch.Tensor:
        input_ids = inputs[0].to(self._device)
        attention_mask = inputs[1].to(self._device)
        return self._network(input_ids, attention_mask)
