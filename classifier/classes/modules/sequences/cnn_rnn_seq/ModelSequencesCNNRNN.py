from typing import Dict

from classifier.classes.core.Model import Model
from classifier.classes.modules.sequences.cnn_rnn_seq.SequencesCNNRNN import SequencesCNNRNN


class ModelSequencesCNNRNN(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = SequencesCNNRNN(network_params).to(self._device)
