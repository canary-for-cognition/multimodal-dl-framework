import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_paths: list, labels: list, loader: callable):
        """
        @param data_paths: list of paths to sequences
        @param labels: list of corresponding labels
        @param loader: transform to be applied on a data item
        """
        self.__data_paths = data_paths
        self.__labels = labels
        self.__loader = loader

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.__data_paths[idx], self.__labels[idx]
        return self.__loader(x), y
