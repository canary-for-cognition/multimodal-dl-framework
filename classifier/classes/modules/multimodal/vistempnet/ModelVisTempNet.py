import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.multimodal.vistempnet.VisTempNet import VisTempNet


class ModelVisTempNet(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        self._send_network_to_device(VisTempNet(network_params))

    def predict(self, inputs: dict) -> torch.Tensor:
        sequences = inputs["sequences"].float().to(self._device)
        images = inputs["images"].float().to(self._device)
        return self._network(sequences, images)
