from classifier.classes.core.Model import Model
from classifier.classes.modules.images.images_cnn.ModelImagesCNN import ModelImagesCNN
from classifier.classes.modules.images.images_cnn_rnn.ModelImagesCNNRNN import ModelImagesCNNRNN
from classifier.classes.modules.images.pre_trained_cnn.ModelPreTrainedCNN import ModelPreTrainedCNN
from classifier.classes.modules.multimodal.vistempnet.ModelVisTempNet import ModelVisTempNet
from classifier.classes.modules.multimodal.vistextnet.ModelVisTextNet import ModelVisTextNet
from classifier.classes.modules.sequences.rnn.ModelRNN import ModelRNN
from classifier.classes.modules.sequences.sequences_cnn.ModelSequencesCNN import ModelSequencesCNN
from classifier.classes.modules.sequences.sequences_cnn_rnn.ModelSequencesCNNRNN import ModelSequencesCNNRNN
from classifier.classes.modules.text.han.ModelHAN import ModelHAN
from classifier.classes.modules.text.transformer.ModelTransformer import ModelTransformer


class ModelFactory:
    models_map = {
        "vistextnet": ModelVisTextNet,
        "vistempnet": ModelVisTempNet,
        "rnn": ModelRNN,
        "sequences_cnn": ModelSequencesCNN,
        "sequences_cnn_rnn": ModelSequencesCNNRNN,
        "images_cnn": ModelImagesCNN,
        "images_cnn_rnn": ModelImagesCNNRNN,
        "pre_trained_cnn": ModelPreTrainedCNN,
        "han": ModelHAN,
        "transformer": ModelTransformer,
    }

    def get(self, model_type: str, model_params: dict) -> Model:
        if model_type not in self.models_map.keys():
            raise ValueError("Model {} is not implemented!".format(model_type))
        return self.models_map[model_type](model_params)
