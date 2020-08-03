import torch
import torch.nn as nn

from classifier.classes.factories.NetworkFactory import NetworkFactory
from classifier.classes.modules.base.MultimodalNN import MultimodalNN


class VisTextNet(MultimodalNN):

    def __init__(self, network_params: dict, activation: bool = True):
        super().__init__(network_params["features_fusion"], activation)

        text_params = network_params["submodules"]["text"]
        text_architecture = text_params["architecture"]
        self.__pre_trained_text_model = text_architecture in ["transformer"]

        images_params = network_params["submodules"]["images"]
        images_architecture = images_params["architecture"]
        self.__stateful_image_model = images_architecture in ["images_cnn_rnn"]

        activate_submodule = self._features_fusion != "early"
        self.images_network = NetworkFactory().get(images_architecture, images_params, activate_submodule)
        self.text_network = NetworkFactory().get(text_architecture, text_params, activate_submodule)

        if self._activation:
            output_size = network_params["output_size"]
            linear_size = network_params["layers"]["linear_1"]["out_features"]
            text_hidden_size = self.text_network.get_hidden_size()
            images_pre_activation_size = self.images_network.get_pre_activation_size()
            self._classifier = nn.Sequential(
                nn.Linear(text_hidden_size + images_pre_activation_size, linear_size),
                nn.ReLU(),
                nn.Linear(linear_size, output_size),
            )

    def forward(self, text: torch.Tensor, images: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward step
        :param text: an embedded text input
        :param images: an image input
        :param attention_mask: the attention mask for the pre-trained models
        :return: the logit for the prediction and the hidden states for both text and images
        """
        x1 = self.text_network(text, attention_mask) if self.__pre_trained_text_model else self.text_network(text)
        x2 = self.images_network(images)
        return self._fuse_features(x1, x2)

    def init_state(self, batch_size: int) -> tuple:
        text_state = self.text_network.init_state(batch_size) if not self.__pre_trained_text_model else None
        img_state = self.images_network.init_state(batch_size) if self.__stateful_image_model else None
        return text_state, img_state

    def using_pre_trained_text_model(self) -> bool:
        return self.__pre_trained_text_model
