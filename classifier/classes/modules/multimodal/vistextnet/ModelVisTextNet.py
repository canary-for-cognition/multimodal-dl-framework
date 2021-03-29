from typing import Dict

import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.multimodal.vistextnet.VisTextNet import VisTextNet


class ModelVisTextNet(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = VisTextNet(network_params).to(self._device)

    def predict(self, inputs: Dict, **kwargs) -> torch.Tensor:
        """
        Performs the preds using VisTextNet
        :param inputs: a dictionary containing "images" and "text" inputs
        :return: the logits of the preds
        """
        images = inputs["images"].to(self._device)
        text = inputs["text"]

        if self._network.using_pre_trained_text_model():
            attention_mask = text[1].to(self._device)
            text = text[0].to(self._device)
        else:
            attention_mask = None
            text.long().to(self._device)

        return self._network(text, images, attention_mask)
