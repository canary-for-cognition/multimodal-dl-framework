import torch
from torch import optim
from transformers import AdamW


class OptimizerFactory:

    def __init__(self, network_parameters: list, learning_rate: float):
        self.optimizers_map = {
            "Adam": optim.Adam(network_parameters, lr=learning_rate),
            "AdamW": AdamW(network_parameters, lr=learning_rate, correct_bias=False),
            "SGD": optim.SGD(network_parameters, lr=learning_rate, momentum=0.9, nesterov=True)
        }

    def get(self, optimizer_type: str) -> torch.optim:
        if optimizer_type not in self.optimizers_map.keys():
            raise ValueError("Optimizer for {} is not implemented!".format(optimizer_type))
        return self.optimizers_map[optimizer_type]
