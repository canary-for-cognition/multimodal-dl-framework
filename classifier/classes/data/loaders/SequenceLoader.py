import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

from classifier.classes.data.loaders.Loader import Loader


class SequenceLoader(Loader):

    def __init__(self, for_submodule: bool = False):
        super().__init__("sequences", for_submodule)

        self.__data_source = self._modality_params["data_source"]
        self.__data_type = self._modality_params["type"]
        self.__max_sequence_length = self._modality_params["length"]
        self.__apply_normalization = self._modality_params["normalize"]
        self.__num_features = self._modality_params["num_features"]
        self.__truncate_from = self._modality_params["truncate_from"]
        self.__truncation_offset = self._modality_params["truncation_offset"]

    def __pad_sequences(self, sequences: np.array) -> np.array:
        """
        Pads the sequences to match the maximum sequence length
        :param sequences: the sequences involving only the selected features
        :return:
        """
        padding = np.zeros((self.__max_sequence_length - len(sequences), self.__num_features))
        return np.append(padding, sequences, axis=0)

    def __truncate(self, sequence: np.array) -> np.array:
        """
        Truncates the sequences according to two axis:
            1. Time steps: according to the truncation starting point (i.e. head or tail) and offset. In case the
                target sequence length is greater than the actual sequence length, the full sequence will be preserved
            2. Features: according to the selected number of features
        :param sequence: the sequence to be truncated
        :return: the truncated sequence
        """
        seq_length = self.__truncation_offset + self.__max_sequence_length
        truncations_map = {
            "head": sequence[self.__truncation_offset:seq_length, :],
            "tail": sequence[-seq_length:-self.__truncation_offset if self.__truncation_offset else None, :]
        }
        truncated_sequence = truncations_map[self.__truncate_from]
        return truncated_sequence[:, :self.__num_features].astype(float)

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads the pickle sequence items and makes it fixed length by removing samples from the
        beginning of the sequence (the oldest) if necessary
        :param path_to_input: the path to the data item to be loaded referred to the main modality
        :return: the fully processed data item
        """
        path_to_item = self._get_path_to_item(path_to_input, self.__data_source, self.__data_type)
        sequence = pickle.load(open(path_to_item, 'rb')) if self._file_format == "pkl" else pd.read_csv(path_to_item)
        sequence = self.__truncate(sequence.values, )

        if len(sequence) < self.__max_sequence_length:
            sequence = self.__pad_sequences(sequence)

        if self.__apply_normalization:
            sequence = normalize(sequence, norm="l2")

        return torch.from_numpy(sequence)
