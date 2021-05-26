from typing import Callable

from classifier.classes.binders.ModalityBinder import ModalityBinder
from classifier.classes.data.loaders.ImageLoader import ImageLoader
from classifier.classes.data.loaders.MultimodalLoader import MultimodalLoader
from classifier.classes.data.loaders.SequenceLoader import SequenceLoader
from classifier.classes.data.loaders.TextLoader import TextLoader


class LoaderFactory:
    loaders_map = {
        "sequences": SequenceLoader,
        "images": ImageLoader,
        "text": TextLoader,
        "multimodal": MultimodalLoader
    }

    def get(self, network_type: str) -> Callable:
        modality = ModalityBinder().get(network_type)
        if isinstance(modality, tuple):
            modality = "multimodal"
        if modality not in self.loaders_map.keys():
            raise ValueError("Loader for {} is not implemented! \n Implemented loaders are: {}"
                             .format(modality, list(self.loaders_map.keys())))
        return self.loaders_map[modality]().load
