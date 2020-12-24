import torch
from torch import nn


class CriterionFactory:
    criteria_map = {"NLLLoss": nn.NLLLoss(), "CrossEntropyLoss": nn.CrossEntropyLoss()}

    def get(self, criterion_type: str) -> torch.nn.modules.loss:
        if criterion_type not in self.criteria_map.keys():
            raise ValueError("Criterion for {} is not implemented!".format(criterion_type))
        return self.criteria_map[criterion_type]
