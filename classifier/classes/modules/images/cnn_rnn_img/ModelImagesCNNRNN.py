from typing import Dict

from classifier.classes.core.Model import Model
from classifier.classes.modules.images.cnn_rnn_img.ImagesCNNRNN import ImagesCNNRNN


class ModelImagesCNNRNN(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = ImagesCNNRNN(network_params).to(self._device)
