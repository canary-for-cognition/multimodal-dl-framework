from typing import Dict

from classifier.classes.data.loaders.ImageLoader import ImageLoader
from classifier.classes.data.loaders.SequenceLoader import SequenceLoader
from classifier.classes.data.loaders.TextLoader import TextLoader
from classifier.classes.utils.Params import Params


class MultimodalLoader:

    def __init__(self):
        network_params = Params.load_network_params(Params.load_experiment_params()["train"]["network_type"])
        self.__modalities = list(network_params["submodules"].keys())

    loaders_map = {"images": ImageLoader, "sequences": SequenceLoader, "text": TextLoader}

    def load(self, path_to_input: str) -> Dict:
        """
        Processes the data items related to the modalities handled by the selected multimodal network
        Example: if VisTextNet is selected, returns images and text
        :param path_to_input: the path to the data item to be loaded (related to the main modality)
        :return: the fully processed data items
        """
        data = {}
        for modality in self.__modalities:
            data[modality] = self.loaders_map[modality](for_submodule=True).load(path_to_input)
        return data
