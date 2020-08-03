from torch import nn

from classifier.classes.modules.images.images_cnn.ImagesCNN import ImagesCNN
from classifier.classes.modules.images.images_cnn_rnn.ImagesCNNRNN import ImagesCNNRNN
from classifier.classes.modules.images.pre_trained_cnn.PreTrainedCNN import PreTrainedCNN
from classifier.classes.modules.sequences.rnn.RNN import RNN
from classifier.classes.modules.sequences.sequences_cnn.SequencesCNN import SequencesCNN
from classifier.classes.modules.sequences.sequences_cnn_rnn.SequencesCNNRNN import SequencesCNNRNN
from classifier.classes.modules.text.han.HAN import HAN
from classifier.classes.modules.text.transformer.Transformer import Transformer


class NetworkFactory:
    networks_map = {
        "rnn": RNN,
        "images_cnn": ImagesCNN,
        "sequences_cnn": SequencesCNN,
        "images_cnn_rnn": ImagesCNNRNN,
        "sequences_cnn_rnn": SequencesCNNRNN,
        "pre_trained_cnn": PreTrainedCNN,
        "han": HAN,
        "transformer": Transformer
    }

    def get(self, network_type: str, module_params: dict, activation: bool = False) -> nn.Module:
        if network_type not in self.networks_map.keys():
            raise ValueError("Network {} is not implemented, could not fetch nn.Module!".format(network_type))
        return self.networks_map[network_type](module_params, activation)
