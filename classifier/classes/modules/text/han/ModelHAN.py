import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.text.han.HAN import HAN


class ModelHAN(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        self._send_network_to_device(HAN(network_params))

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        inputs = inputs.long().to(self._device)
        return self._network(inputs)
