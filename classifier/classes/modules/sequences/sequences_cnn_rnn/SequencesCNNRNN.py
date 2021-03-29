from typing import Dict

from classifier.classes.modules.base.CNNRNN import CNNRNN
from classifier.classes.modules.sequences.sequences_cnn.SequencesCNN import SequencesCNN


class SequencesCNNRNN(CNNRNN):

    def __init__(self, network_params: Dict, activate: bool = True):
        super().__init__(network_params, activate)

        cnn_params = network_params["cnn"]
        cnn_params["modality"] = network_params["modality"]
        cnn_params["layers"]["classifier"]["linear_1"]["out_features"] = self.__hidden_size

        self.cnn = SequencesCNN(cnn_params, activation=False)
