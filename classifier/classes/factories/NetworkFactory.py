from typing import Dict

from torch import nn

from classifier.classes.modules.images.cnn_img.ImagesCNN import ImagesCNN
from classifier.classes.modules.images.cnn_rnn_img.ImagesCNNRNN import ImagesCNNRNN
from classifier.classes.modules.images.pre_trained_cnn.PreTrainedCNN import PreTrainedCNN
from classifier.classes.modules.sequences.cnn_rnn_seq.SequencesCNNRNN import SequencesCNNRNN
from classifier.classes.modules.sequences.cnn_seq.SequencesCNN import SequencesCNN
from classifier.classes.modules.sequences.rnn.RNN import RNN
from classifier.classes.modules.text.transformer.Transformer import Transformer


class NetworkFactory:
    networks_map = {
        "rnn": RNN,
        "cnn_img": ImagesCNN,
        "cnn_seq": SequencesCNN,
        "cnn_rnn_img": ImagesCNNRNN,
        "cnn_rnn_seq": SequencesCNNRNN,
        "pre_trained_cnn": PreTrainedCNN,
        "transformer": Transformer
    }

    def get(self, network_type: str, module_params: Dict, activation: bool = False) -> nn.Module:
        if network_type not in self.networks_map.keys():
            raise ValueError("Network {} is not implemented! \n Implemented networks are: {}"
                             .format(network_type, list(self.networks_map.keys())))
        return self.networks_map[network_type](module_params, activation)
