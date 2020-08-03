from classifier.classes.core.Model import Model

from classifier.classes.modules.images.images_cnn.ImagesCNN import ImagesCNN


class ModelImagesCNN(Model):

    def __init__(self, network_params: dict):
        super().__init__(device=network_params["device"])
        self._network = ImagesCNN(network_params).float().to(self._device)
