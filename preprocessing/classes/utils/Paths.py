import os
from typing import Union

from preprocessing.classes.utils.Params import Params


class Paths:

    def __init__(self):
        params = Params.load_dataset_params()

        self.__dataset_type = params["dataset_name"]
        self.__labels = params["labels"]

        path_to_dataset = os.path.join("..", "dataset", self.__dataset_type)

        self.__path_to_dataset_modalities = os.path.join(path_to_dataset, "modalities")
        self.__path_to_preprocessed_modalities = os.path.join(params["paths"]["preprocessed_data"], "modalities")

        self.__path_to_tasks = os.path.join(params["paths"]["preprocessed_data"], "tasks")

        self.__path_to_metadata = os.path.join(path_to_dataset, "metadata")

        self.__modality_sources = {
            "dataset": self.__path_to_dataset_modalities,
            "preprocessed": self.__path_to_preprocessed_modalities
        }

    def get_labels(self) -> dict:
        return self.__labels

    def get_dataset_type(self) -> str:
        return self.__dataset_type

    def get_metadata(self, metadata_type: str) -> str:
        return os.path.join(self.__path_to_metadata, metadata_type)

    def get_paths_to_tasks(self) -> dict:
        tasks = ["cookie_theft"]
        return {task: self.__create_paths(os.path.join(self.__path_to_tasks, task)) for task in tasks}

    def get_paths_to_modality(self, path_components: dict, return_base_path: bool = False) -> Union[str, dict]:
        """
        Returns the paths to the labels at "modality"/"data_source"/"representation"/ (e.g. "images/audio/mfcc")
        or the networks path to "modality"/"data_source"/"representation" depending on "return_base_path"
        :param path_components: a dict containing:
            * data_folder: the source of the modality to be considered ("dataset" or "preprocessed")
            * modality: the modality to be considered (e.g. "sequences", "images", etc...)
            * data_source: the source of the data for the selected modality (e.g. "eye_tracking", "audio", etc...)
            * representation: the data representation type for the selected modality (e.g. "heatmaps" for "images")
            * data_dimension: the data_dimension of the data (i.e. "raw", "base", "augmented")
        :param return_base_path: whether or not to return the networks path only
        :return: a dictionary containing the paths to the labels at "modality"/"data_source"/"representation" or
                the base path to "modality"/"data_source"/"representation" depending on "return_base_path"
        """
        data_folder = path_components["data_folder"]

        path_to_modality = os.path.join(self.__modality_sources[data_folder],
                                        path_components["modality"],
                                        path_components["data_source"],
                                        path_components["representation"],
                                        path_components["data_dimension"])

        return path_to_modality if return_base_path else self.__create_paths(path_to_modality, data_folder)

    def __create_paths(self, base_path: str, data_folder: str = "preprocessed") -> dict:
        paths = {
            "pos": os.path.join(base_path, self.__labels["positive"]),
            "neg": os.path.join(base_path, self.__labels["negative"])
        }

        if data_folder != "dataset":
            os.makedirs(paths["pos"], exist_ok=True)
            os.makedirs(paths["neg"], exist_ok=True)

        return paths
