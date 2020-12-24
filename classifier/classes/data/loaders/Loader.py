import os

import torch

from classifier.classes.utils.Params import Params


class Loader:

    def __init__(self, modality: str, for_submodule: bool = False):
        self._modality = modality
        self._modality_params = Params.load_modality_params(self._modality)

        experiment_params = Params.load_experiment_params()
        dataset_params = Params.load_dataset_params(experiment_params["dataset_name"])
        self._path_to_modalities = dataset_params["paths"]
        self._network_type = experiment_params["train"]["network_type"]

        if for_submodule:
            multimodal_network_params = Params.load_network_params(self._network_type)
            self._network_type = multimodal_network_params["submodules"][self._modality]["architecture"]

        path_to_modality = self._path_to_modalities[self._modality]
        self._path_to_data = os.path.join(path_to_modality, self._modality_params["path_to_data"])
        self._file_format = self._modality_params["file_format"]

    def _get_path_to_item(self, path_to_input: str) -> str:
        """
        Creates the path to the data item for the specified modality
        :param path_to_input: the path to the data item related to the main modality
        :return: the path to the eye-tracking sequence data item
        """
        split_path = path_to_input.split(os.sep)
        file_name = str(split_path[-1]).split(".")[0] + "." + self._file_format
        label = str(split_path[-2])
        return os.path.join(self._path_to_data, label, file_name)

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads a data item from the dataset
        :param path_to_input: the path to the data item to be loaded (referred to the main modality)
        :return: the loaded data item
        """
        pass
