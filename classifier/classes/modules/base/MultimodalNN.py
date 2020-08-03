import torch
from torch import nn


class MultimodalNN(nn.Module):
    def __init__(self, features_fusion: str, activation: bool):
        super().__init__()

        self._features_fusion = features_fusion
        self._activation = activation if features_fusion == "early" else False

        if self._activation:
            self._classifier = None

        if self._features_fusion == "late_blending":
            self.__classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(4, 2)
            )

    def __early_fuse(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Early fuses two features tensors by concatenating them
        :param x1: the first features tensor to concatenate
        :param x2: the second features tensor to concatenate
        :return: the early fused features tensor
        """
        x = torch.cat((x1, x2), 1)
        return self._classifier(x) if self._activation else x

    @staticmethod
    def __late_average_voting_fuse(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Late fuses two logits tensors by ensemble averaging them
        :param x1: the first logits tensor to fuse
        :param x2: the second logits tensor to fuse
        :return: the late fused with average voting logits tensor
        """
        return torch.mean(torch.stack([x1, x2]), dim=0)

    def __late_blending_fuse(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Late fuses two logits tensors by feeding them as input to a classifier
        :param x1: the first logits tensor to fuse
        :param x2: the second logits tensor to fuse
        :return: the late fused with blending logits tensor
        """
        return self.__classifier(torch.cat((x1, x2), 1))

    def _fuse_features(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Late fuses two logits tensors by feeding them as input to a classifier
        :param x1: the first logits tensor to fuse
        :param x2: the second logits tensor to fuse
        :return: the fused logits tensor
        """
        features_fusions_map = {
            "early": self.__early_fuse,
            "late_average_voting": self.__late_average_voting_fuse,
            "late_blending": self.__late_blending_fuse,
        }
        return features_fusions_map[self._features_fusion](x1, x2)
