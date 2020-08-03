import torch

from classifier.classes.core.Model import Model
from classifier.classes.modules.multimodal.vistextnet.VisTextNet import VisTextNet


class ModelVisTextNet(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        self._network = VisTextNet(network_params).float().to(self._device)

    def predict(self, inputs: dict, **kwargs) -> torch.Tensor:
        """
        Performs the predictions using VisTextNet
        :param inputs: a dictionary containing "images" and "text" inputs
        :return: the logits of the predictions
        """
        images = inputs["images"].float().to(self._device)
        text = inputs["text"]

        if self._network.using_pre_trained_text_model():
            attention_mask = text[1].to(self._device)
            text = text[0].to(self._device)
        else:
            attention_mask = None
            text.long().to(self._device)

        return self._network(text, images, attention_mask)
