import os
from abc import ABC
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import DatasetFolder

from classifier.classes.factories.GrouperFactory import GrouperFactory
from classifier.classes.utils.Params import Params


class Dataset(torch.utils.data.Dataset, ABC):

    def __init__(self, params: dict, batch_size: int, loader: Callable):
        """
        :param params: the dataset params stored in the configuration file
        :param batch_size: the size of the batch of data to be fed to the model
        :param loader: the loader function for the selected architecture
        """
        self.__dataset_type = params["type"]

        self.__neg_class = "0_" + params["classes"]["neg"]
        self.__pos_class = "1_" + params["classes"]["pos"]

        self.__loader = loader
        self.__batch_size = batch_size
        self.__path_to_dataset_folder = params["paths"]["dataset_folder"]
        self.__path_to_dataset_metadata = params["paths"]["dataset_metadata"]

        main_modality = params["main_modality"]
        path_to_main_modality = params["paths"][main_modality]
        main_modality_info = self.__fetch_main_modality_info(main_modality, path_to_main_modality)
        self.__augmented = main_modality_info["augmented"]
        self.__path_to_main_modality = main_modality_info["path"]
        self.__file_format = main_modality_info["file_format"]

        self.__data = self.create_dataset(self.__path_to_main_modality)

    def __len__(self) -> int:
        return len(self.__data)

    def get_data(self) -> pd.DataFrame:
        return self.__data

    def get_dataset_type(self) -> str:
        return self.__dataset_type

    def get_path_to_dataset_folder(self) -> str:
        return self.__path_to_dataset_folder

    def get_path_to_main_modality(self) -> str:
        return self.__path_to_main_modality

    def get_file_format(self) -> str:
        return self.__file_format

    def get_classes(self) -> tuple:
        return self.__neg_class, self.__pos_class

    def is_augmented(self) -> bool:
        return self.__augmented

    def print_data_loader(self, data_loader: DataLoader, percentage: float, data_type: str):
        """
        Prints an overview of the given data_loader for the current data split
        :param data_loader: a PyTorch data_loader
        :param percentage: the percentage of the split represented by the data_loader
        :param data_type: the type of incoming data ("train", "val" or "test")
        """
        dataset_size = len(data_loader.dataset)
        num_pos = int(sum([sum(labels) for _, (inputs, labels) in enumerate(data_loader)]))
        print(" {} set ({:.2f}%): \n\n".format(data_type.capitalize(), percentage))
        print("\t - Total number of items: {} \n".format(dataset_size))
        print("\t - {}: {} \n".format(self.__pos_class, num_pos))
        print("\t - {}: {} \n".format(self.__pos_class, dataset_size - num_pos))

    @staticmethod
    def __fetch_main_modality_info(modality: str, path_to_modality: str) -> dict:
        """
        Fetches information about the main modality from the related JSON file
        :param modality: the main modality of data which the dataset will be be based on
        :param path_to_modality: the path to the folder containing the main modality of data
        :return: a dict containing a flag stating whether the modality requires augmentation or not,
                 the path to the files of the main modality and the specification of their format
        """
        main_modality_params = Params.load_modality_params(modality)
        data_source = main_modality_params["data_source"] if "data_source" in main_modality_params.keys() else ""
        representation_type = main_modality_params["type"] if "type" in main_modality_params.keys() else ""
        path_to_modality = os.path.join(path_to_modality, data_source, representation_type)
        augmented = False

        if "augment" in main_modality_params.keys():
            augmented = main_modality_params["augment"]
            path_to_modality = os.path.join(path_to_modality, "augmented" if augmented else "base")

        file_format = "." + main_modality_params["file_format"]

        return {"augmented": augmented, "path": path_to_modality, "file_format": file_format}

    def __data_from_filename(self, filenames: list) -> list:
        """
        Fetches metadata from the names of the files
        :param filenames: the name of the files containing serialized data items
        :return: a list of lists of data extracted from the names of the files
        """
        group_info_extractor = GrouperFactory().get_group_info_extractor(self.__dataset_type)
        data = []
        for filename in filenames:
            item_info = group_info_extractor(filename)
            data.append([filename, item_info["id"], item_info["group"]])
        return data

    def create_dataset(self, path_to_main_modality: str) -> pd.DataFrame:
        """
        Creates a pd.DataFrame and related CSV file containing the information about the dataset.
        That is, the list of its items, related ids and groups
        :param path_to_main_modality: the path to the files of the main modality
        :return: a pd.DataFrame containing the information about the dataset
        """
        path_to_neg = os.path.join(path_to_main_modality, self.__neg_class)
        path_to_pos = os.path.join(path_to_main_modality, self.__pos_class)
        neg_data = self.__data_from_filename(os.listdir(path_to_neg))
        pos_data = self.__data_from_filename(os.listdir(path_to_pos))
        dataset = pd.DataFrame(neg_data + pos_data, columns=["file_name", "item_id", "group"])

        paths_to_neg = [os.path.join(path_to_neg, item) for item in os.listdir(path_to_neg)]
        paths_to_pos = [os.path.join(path_to_pos, item) for item in os.listdir(path_to_pos)]
        labels = np.concatenate((np.zeros(len(paths_to_neg)), np.ones(len(paths_to_pos))))
        dataset = dataset.join(pd.DataFrame({"path": paths_to_neg + paths_to_pos, "label": labels}))

        dataset.to_csv(os.path.join(self.__path_to_dataset_metadata, "dataset.csv"), index=False)

        return dataset

    def create_data_loader(self, data: DatasetFolder, shuffle: bool = False) -> DataLoader:
        """
        Creates a data loader for the given data for the established parameters
        :param data: a PyTorch data folder
        :param shuffle: whether or not to shuffle the data
        :return: a data loader for the given data and the established parameters
        """
        return DataLoader(data, self.__batch_size, shuffle=shuffle, num_workers=16, pin_memory=True, drop_last=False)

    def create_dataset_folder(self, path_to_folder: str) -> DatasetFolder:
        """
        Creates a data folder for the given path
        :param path_to_folder: the path to the folder containing the data split by class
        :return: a dataset folder for the data at the given path
        """
        return datasets.DatasetFolder(path_to_folder, loader=self.__loader, extensions=tuple(self.__file_format))
