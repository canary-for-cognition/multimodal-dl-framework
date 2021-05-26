from typing import Dict

from classifier.classes.core.Model import Model
from classifier.classes.modules.images.cnn_img.ModelImagesCNN import ModelImagesCNN
from classifier.classes.modules.images.cnn_rnn_img.ModelImagesCNNRNN import ModelImagesCNNRNN
from classifier.classes.modules.images.pre_trained_cnn.ModelPreTrainedCNN import ModelPreTrainedCNN
from classifier.classes.modules.multimodal.vistempnet.ModelVisTempNet import ModelVisTempNet
from classifier.classes.modules.multimodal.vistextnet.ModelVisTextNet import ModelVisTextNet
from classifier.classes.modules.sequences.cnn_rnn_seq.ModelSequencesCNNRNN import ModelSequencesCNNRNN
from classifier.classes.modules.sequences.cnn_seq.ModelSequencesCNN import ModelSequencesCNN
from classifier.classes.modules.sequences.rnn.ModelRNN import ModelRNN
from classifier.classes.modules.text.transformer.ModelTransformer import ModelTransformer


class ModelFactory:
    models_map = {
        "vistextnet": ModelVisTextNet,
        "vistempnet": ModelVisTempNet,
        "rnn": ModelRNN,
        "cnn_seq": ModelSequencesCNN,
        "cnn_rnn_seq": ModelSequencesCNNRNN,
        "cnn_img": ModelImagesCNN,
        "cnn_rnn_img": ModelImagesCNNRNN,
        "pre_trained_cnn": ModelPreTrainedCNN,
        "transformer": ModelTransformer,
    }

    def get(self, model_type: str, model_params: Dict) -> Model:
        if model_type not in self.models_map.keys():
            raise ValueError("Model {} is not implemented! \n Implemented models are: {}"
                             .format(model_type, list(self.models_map.keys())))
        return self.models_map[model_type](model_params)
