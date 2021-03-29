from typing import Dict

from classifier.classes.core.Model import Model
from classifier.classes.modules.images.pre_trained_cnn.PreTrainedCNN import PreTrainedCNN


class ModelPreTrainedCNN(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = PreTrainedCNN(network_params).to(self._device)
