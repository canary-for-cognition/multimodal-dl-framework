from typing import Dict

from classifier.classes.modules.base.CNNRNN import CNNRNN
from classifier.classes.modules.images.cnn_img.ImagesCNN import ImagesCNN


class ImagesCNNRNN(CNNRNN):

    def __init__(self, network_params: Dict, activate: bool = True):
        super().__init__(network_params, activate)

        cnn_params = network_params["cnn"]
        cnn_params["modality"] = network_params["modality"]
        cnn_params["layers"]["classifier"]["linear_1"]["out_features"] = self.__hidden_size

        self.cnn = ImagesCNN(cnn_params, activation=False)
