import os

import torch

from classifier.classes.utils.Params import Params


class Loader:

    def __init__(self, modality: str, for_submodule: bool = False):
        self._modality = modality
        self._modality_params = Params.load_modality_params(self._modality)

        experiment_params = Params.load_experiment_params()
        self._dataset_params = Params.load_dataset_params(experiment_params["dataset_type"])

        self._network_type = experiment_params["network_type"]

        if for_submodule:
            multimodal_network_params = Params.load_network_params(self._network_type)
            self._network_type = multimodal_network_params["submodules"][self._modality]["architecture"]

        self._network_params = Params.load_network_params(self._network_type)

        self._path_to_modality = self._dataset_params["paths"][self._modality]
        self._file_format = self._modality_params["file_format"]
        self._augmented = self._modality_params["augment"]
        self._data_folder = "augmented" if self._augmented else "base"

    def _get_path_to_item(self, path_to_input: str, data_source: str = "", data_type: str = "") -> str:
        """
        Creates the path to the data item for the specified modality
        :param path_to_input: the path to the data item related to the main modality
        :param data_source: the source of the data (e.g. "eye_tracking", etc...)
        :return: the path to the eye-tracking sequence data item
        """
        split_path = path_to_input.split(os.sep)
        label = str(split_path[-2])
        file_name = str(split_path[-1]).split(".")[0] + "." + self._file_format
        return os.path.join(self._path_to_modality, data_source, data_type, self._data_folder, label, file_name)

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads a data item from the dataset
        :param path_to_input: the path to the data item to be loaded (referred to the main modality)
        :return: the loaded data item
        """
        pass
