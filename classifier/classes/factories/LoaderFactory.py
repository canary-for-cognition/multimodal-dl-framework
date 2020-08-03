from classifier.classes.data.loaders.ImageLoader import ImageLoader
from classifier.classes.data.loaders.MultimodalLoader import MultimodalLoader
from classifier.classes.data.loaders.SequenceLoader import SequenceLoader
from classifier.classes.data.loaders.TextLoader import TextLoader
from classifier.classes.binders.ModalityBinder import ModalityBinder


class LoaderFactory:
    loaders_map = {
        "sequences": SequenceLoader,
        "images": ImageLoader,
        "text": TextLoader,
        "multimodal": MultimodalLoader
    }

    def get(self, network_type: str) -> callable:
        modality = ModalityBinder().get(network_type)
        if isinstance(modality, tuple):
            modality = "multimodal"
        if modality not in self.loaders_map.keys():
            raise ValueError("Loader for {} is not implemented!".format(modality))
        return self.loaders_map[modality]().load
