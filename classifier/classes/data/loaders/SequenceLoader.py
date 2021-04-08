import numpy as np
import pandas as pd
import pickle5 as pickle
import torch

from classifier.classes.data.loaders.Loader import Loader


class SequenceLoader(Loader):

    def __init__(self, for_submodule: bool = False):
        super().__init__("sequences", for_submodule)
        self.__max_seq_len = self._modality_params["length"]
        self.__num_features = self._modality_params["num_features"]
        self.__truncate_from = self._modality_params["truncate_from"]

    def __pad_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Pads the sequences to match the maximum sequence length
        :param sequences: the sequences involving only the selected features
        :return:
        """
        padding = np.zeros((self.__max_seq_len - len(sequences), self.__num_features))
        return np.append(padding, sequences, axis=0)

    def __truncate(self, sequence: np.ndarray) -> np.ndarray:
        """
        Truncates the sequences according to two axis:
            1. Time steps: according to the truncation starting point (i.e. head or tail) and offset. In case the
                target sequence length is greater than the actual sequence length, the full sequence will be preserved
            2. Features: according to the selected number of features
        :param sequence: the sequence to be truncated
        :return: the truncated sequence
        """
        truncations_map = {
            "head": sequence[:self.__max_seq_len, :],
            "tail": sequence[-self.__max_seq_len:, :]
        }
        return truncations_map[self.__truncate_from][:, :self.__num_features]

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads the pickle sequence items and makes it fixed length by removing samples from the
        beginning of the sequence (the oldest) if necessary
        :param path_to_input: the path to the data item to be loaded referred to the main modality
        :return: the fully processed data item
        """
        path_to_item = self._get_path_to_item(path_to_input)
        sequence = pickle.load(open(path_to_item, 'rb')) if self._file_format == "pkl" else pd.read_csv(path_to_item)
        sequence = self.__truncate(sequence.values)

        if len(sequence) < self.__max_seq_len:
            sequence = self.__pad_sequences(sequence)

        return torch.from_numpy(sequence.astype(np.float32))
