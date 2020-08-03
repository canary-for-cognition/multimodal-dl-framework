import multiprocessing as mp
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torchvision import datasets

from classifier.classes.factories.GrouperFactory import GrouperFactory
from classifier.classes.utils.Params import Params


class Dataset(torch.utils.data.Dataset):

    def __init__(self, params: dict, batch_size: int, loader: Callable, device: torch.device):
        """
        :param params: the dataset params stored in the configuration file
        :param batch_size: the size of the batch of data to be fed to the model
        :param loader: the loader function for the selected architecture
        :param device: the device which to run on (gpu or cpu)
        """
        self.__dataset_type = params["type"]
        self.__variable_size_input = params["variable_size_input"]

        self.__negative_class = "0_" + params["classes"]["negative"]
        self.__positive_class = "1_" + params["classes"]["positive"]

        self.__device = device
        self.__loader = loader
        self.__batch_size = batch_size
        self.__path_to_dataset_folder = params["paths"]["dataset_folder"]
        self.__path_to_dataset_metadata = params["paths"]["dataset_metadata"]

        main_modality = params["main_modality"]
        base_path_to_main_modality = params["paths"][main_modality]
        main_modality_info = self.__fetch_main_modality_info(main_modality, base_path_to_main_modality)
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
        return self.__negative_class, self.__positive_class

    def is_augmented(self) -> bool:
        return self.__augmented

    def print_data_loader(self, data_loader: torch.utils.data.DataLoader, percentage: float, data_type: str):
        """
        Prints an overview of the given data_loader for the current data split
        :param data_loader: a PyTorch data_loader
        :param percentage: the percentage of the split represented by the data_loader
        :param data_type: the type of incoming data ("training", "validation" or "test")
        """
        num_pos = int(sum([sum(labels) for _, (inputs, labels) in enumerate(data_loader)]))
        print(" {t} set ({percentage:.2f}%): \n\n"
              "\t - Total number of elements : {tot} \n"
              "\t - Number of {pos} elements : {n_pos}\n"
              "\t - Number of {neg} elements : {n_neg} \n".format(t=data_type.capitalize(),
                                                                  percentage=percentage,
                                                                  tot=len(data_loader.dataset),
                                                                  neg=self.__negative_class,
                                                                  pos=self.__positive_class,
                                                                  n_pos=num_pos,
                                                                  n_neg=len(data_loader.dataset) - num_pos))

    @staticmethod
    def __fetch_main_modality_info(main_modality: str, base_path_to_main_modality: str) -> dict:
        """
        Fetches information about the main modality from the related JSON file
        :param main_modality: the main modality of data which the dataset will be be based on
        :param base_path_to_main_modality: the path to the folder containing the main modality of data
        :return: a dict containing a flag stating whether the modality requires augmentation or not, the path to the
                 files of the main modality and the specification of their format
        """
        main_modality_params = Params.load_modality_params(main_modality)
        data_source = main_modality_params["data_source"] if "data_source" in main_modality_params.keys() else ""
        representation_type = main_modality_params["type"] if "type" in main_modality_params.keys() else ""
        path_to_main_modality = os.path.join(base_path_to_main_modality, data_source, representation_type)
        augmented = False

        if "augment" in main_modality_params.keys():
            augmented = main_modality_params["augment"]
            path_to_main_modality = os.path.join(path_to_main_modality, "augmented" if augmented else "base")

        file_format = "." + main_modality_params["file_format"]

        return {
            "augmented": augmented,
            "path": path_to_main_modality,
            "file_format": file_format
        }

    def __data_from_filename(self, filenames: list) -> list:
        """
        Fetches the information about the experiments from the names of the files
        :param filenames: the name of the files containing serialized data items
        :return: a list of lists of data extracted from the names of the files
        """
        group_info_extractor = GrouperFactory().get_group_info_extractor(self.__dataset_type)
        data = []
        for filename in filenames:
            item_info = group_info_extractor(filename)
            data.append([filename, item_info["id"], item_info["frame"]])
        return data

    def create_dataset(self, path_to_main_modality: str) -> pd.DataFrame:
        """
        Creates a pd.DataFrame and related CSV file containing the information about the dataset.
        That is, the list of its items, related ids and frame sets (if the main modality requires augmentation)
        :param path_to_main_modality: the path to the files of the main modality
        :return: a pd.DataFrame containing the information about the dataset
        """
        path_to_negative = os.path.join(path_to_main_modality, self.__negative_class)
        path_to_positive = os.path.join(path_to_main_modality, self.__positive_class)

        negative_data = self.__data_from_filename(os.listdir(path_to_negative))
        positive_data = self.__data_from_filename(os.listdir(path_to_positive))
        data = negative_data + positive_data
        dataset = pd.DataFrame(data, columns=["file_name", "item_id", "frame"])

        paths_to_negative = [os.path.join(path_to_negative, item) for item in os.listdir(path_to_negative)]
        paths_to_positive = [os.path.join(path_to_positive, item) for item in os.listdir(path_to_positive)]
        labels = np.concatenate((np.zeros(len(paths_to_negative)), np.ones(len(paths_to_positive))))
        dataset = dataset.join(pd.DataFrame({"path": paths_to_negative + paths_to_positive, "label": labels}))

        path_to_csv = os.path.join(self.__path_to_dataset_metadata, "dataset.csv")
        if os.path.exists(path_to_csv):
            os.remove(path_to_csv)
        dataset.to_csv(path_to_csv, index=False)

        return dataset

    @staticmethod
    def __variable_size_collate(batch: list) -> list:
        data = [torch.transpose(item[0], 0, -1) for item in batch]
        padded_data = torch.transpose(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), 1, -1)
        target = torch.LongTensor([item[1] for item in batch])
        return [padded_data, target]

    def __select_collate(self) -> Callable:
        return self.__variable_size_collate if self.__variable_size_input else None

    def __select_num_workers(self) -> int:
        return 10 if self.__device.type == 'cuda' else mp.cpu_count()

    def create_data_loader(self, data: datasets.DatasetFolder, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """
        Creates a data loader for the given data for the established parameters
        :param data: a PyTorch data folder
        :param shuffle: whether or not to shuffle the data
        :return: a data loader for the given data and the established parameters
        """
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.__batch_size,
                                           shuffle=shuffle,
                                           num_workers=self.__select_num_workers(),
                                           collate_fn=self.__select_collate(),
                                           pin_memory=True,
                                           drop_last=False)

    def create_dataset_folder(self, path_to_folder: str) -> datasets.DatasetFolder:
        """
        Creates a data folder for the given path
        :param path_to_folder: the path to the folder containing the data split by class
        :return: a dataset folder for the data at the given path
        """
        return datasets.DatasetFolder(path_to_folder, loader=self.__loader, extensions=tuple(self.__file_format))
