import torch

from classifier.classes.modules.base.CNN import CNN


class SequencesCNN(CNN):

    def __init__(self, network_params: dict, activation: bool = True):
        network_params["input_size"] = (-1, network_params["modality"]["length"])
        network_params["layers"]["conv_block"]["conv_1"]["in_channels"] = network_params["modality"]["num_features"]
        super().__init__(network_params, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 1))
