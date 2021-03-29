import os
from typing import Dict

from preprocessing.classes.utils.Params import Params


class Paths:

    def __init__(self):
        params = Params.load_dataset_params()
        self.__dataset_type = params["dataset_name"]
        self.__labels = params["labels"]
        path_to_dataset = os.path.join("..", "dataset", self.__dataset_type)
        self.__path_to_modalities = os.path.join(path_to_dataset, "modalities")
        self.__path_to_metadata = os.path.join(path_to_dataset, "metadata")

    def get_labels(self) -> Dict:
        return self.__labels

    def get_dataset_type(self) -> str:
        return self.__dataset_type

    def get_metadata(self, metadata_type: str) -> str:
        return os.path.join(self.__path_to_metadata, metadata_type)

    def create_paths(self, base_path: str) -> Dict:
        paths = {
            "pos": os.path.join(self.__path_to_modalities, base_path, self.__labels["positive"]),
            "neg": os.path.join(self.__path_to_modalities, base_path, self.__labels["negative"])
        }
        os.makedirs(paths["pos"], exist_ok=True)
        os.makedirs(paths["neg"], exist_ok=True)
        return paths
