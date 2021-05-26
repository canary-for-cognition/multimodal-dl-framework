from typing import Dict, Tuple

import torch
import torch.nn as nn

from classifier.classes.factories.NetworkFactory import NetworkFactory
from classifier.classes.modules.base.MultimodalNN import MultimodalNN


class VisTempNet(MultimodalNN):

    def __init__(self, network_params: Dict, activation: bool = True):
        super().__init__(network_params["fusion_policy"], activation)

        images_params = network_params["submodules"]["images"]
        images_params["device"] = network_params["device"]
        images_architecture = images_params["architecture"]
        self.__stateful_image_model = images_architecture not in ["cnn_lstm"]

        sequences_params = network_params["submodules"]["sequences"]
        sequences_params["device"] = network_params["device"]
        sequences_params["input_size"] = sequences_params["modality"]["num_features"]
        sequences_architecture = sequences_params["architecture"]

        activate_submodule = self._fusion_policy != "early"
        self.images_network = NetworkFactory().get(images_architecture, images_params, activate_submodule)
        self.sequences_network = NetworkFactory().get(sequences_architecture, sequences_params, activate_submodule)

        if self._activation:
            output_size = network_params["output_size"]
            linear_size = network_params["layers"]["linear_1"]["out_features"]
            hidden_size = self.sequences_network.get_hidden_size()
            images_linear_size = self.images_network.get_pre_activation_size()

            self._classifier = nn.Sequential(
                nn.Linear(hidden_size + images_linear_size, linear_size),
                nn.ReLU(),
                nn.Linear(linear_size, output_size),
            )

    def forward(self, sequences: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward step
        :param sequences: a batch of temporal sequences [batch_size, sequences_length, num_features]
        :param images: a batch of W x H images of shape [batch_size, num_channels, img_width, img_height]
        :return: the logit for the prediction and the hidden states
        """
        state = self.sequences_network.init_state(sequences.shape[0])
        x1 = self.sequences_network(sequences, state)
        x2 = self.images_network(images)
        return self._fuse_features(x1, x2)

    def set_state(self, seq_state: torch.Tensor, img_state: torch.Tensor = None):
        """
        Sets the current state of the stateful sub networks
        :param img_state: the hidden states (and cell states if using LSTM) for the sub network handling images
        :param seq_state: the hidden states (and cell states if using LSTM) for the sub network handling sequences
        """
        self.sequences_network.set_state(seq_state)
        if self.__stateful_image_model:
            self.images_network.set_state(img_state)

    def init_state(self, batch_size: int) -> Tuple:
        seq_state = self.sequences_network.init_state(batch_size)
        img_state = self.images_network.init_state(batch_size) if self.__stateful_image_model else None
        return seq_state, img_state
