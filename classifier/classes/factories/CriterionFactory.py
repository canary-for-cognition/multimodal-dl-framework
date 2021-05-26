import torch
from torch import nn


class CriterionFactory:
    criteria_map = {"NLLLoss": nn.NLLLoss(), "CrossEntropyLoss": nn.CrossEntropyLoss()}

    def get(self, criterion_type: str) -> torch.nn.modules.loss:
        if criterion_type not in self.criteria_map.keys():
            raise ValueError("Criterion for {} is not implemented! \n Supported criteria are: {}"
                             .format(criterion_type, list(self.criteria_map.keys())))
        return self.criteria_map[criterion_type]
