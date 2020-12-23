import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.multimodal.vistempnet.VisTempNet import VisTempNet


class ModelVisTempNet(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        self._network = VisTempNet(network_params).float().to(self._device)

    def predict(self, x: dict, **kwargs) -> torch.Tensor:
        seq = x["sequences"].to(self._device)
        img = x["images"].to(self._device)
        return self._network(seq, img)
