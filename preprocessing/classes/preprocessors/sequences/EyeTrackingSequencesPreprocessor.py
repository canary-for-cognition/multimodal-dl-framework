import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.classes.base.Preprocessor import Preprocessor
from preprocessing.classes.utils.AssetLoader import AssetLoader


class EyeTrackingSequencesPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__("eye_tracking_sequences")

        self.__validity_left_idx, self.__validity_right_idx = None, None
        self.__left_idxes, self.__right_idxes = None, None

        self.__features = AssetLoader.load_features("eye_tracking_sequences")
        self.__selected_features = self.__select_features(self._params["features"])

        self.__verbose = self._params["verbose"]
        self.__max_sequence_length = self._params["max_sequence_length"]

        self.__filter_by_event = self._params["filter_by_event"]["active"]
        if self.__filter_by_event:
            self.__event_type = self._params["filter_by_event"]["type"]

        self.__use_cyclic_split = self._params["cyclic_split"]["active"]
        if self.__use_cyclic_split:
            self.__split_step = self._params["cyclic_split"]["step"]

        self.__collapse_fixations = self._params["collapse_fixations"]
        self.__fill_na_with_mean = self._params["fill_na_with_mean"]

        self.__paths_to_raw_sequences = self._paths.get_paths_to_modality(self._params["paths"]["source"])
        self.__paths_to_base_sequences = self._paths.get_paths_to_modality(self._params["paths"]["destination"])
        self._params["paths"]["destination"]["data_dimension"] = "augmented"
        self.__paths_to_augmented_sequences = self._paths.get_paths_to_modality(self._params["paths"]["destination"])

    def __update_idxes(self, features: list):
        self.__validity_left_idx = features.index("ValidityLeft")
        self.__validity_right_idx = features.index("ValidityRight")
        self.__left_idxes, self.__right_idxes = [], []
        for i, f in enumerate(features):
            if "Left" in f:
                self.__left_idxes.append(i)
            elif "Right" in f:
                self.__right_idxes.append(i)

    def __select_features(self, selected_features: str) -> list:
        if selected_features == "all":
            features = [f for group in self.__features.keys() for f in self.__features[group]]
        else:
            features = [f for group in selected_features.split("_") for f in self.__features[group]]
            features += self.__features["event"] + self.__features["validity"]
        self.__update_idxes(features)
        return features

    def __percentage_partially_invalid_rows(self, data: np.array) -> float:
        """
        Checks the percentage of invalid rows in a data item where at least one eye is invalid.
        We define a row as invalid if all of the features named below contain invalid values
        :param data: item whose missing eye values will be fixed
        :return the percentage of invalid rows
        """
        data = pd.DataFrame(data)
        total_rows = data.shape[0]
        invalid = data[
            (data.iloc[:, self.__validity_left_idx] == 4.0) | (data.iloc[:, self.__validity_right_idx] == 4.0)]
        num_invalid = invalid.shape[0]
        return (num_invalid / total_rows) * 100

    @staticmethod
    def __fix_eye_side(data: np.array, invalid_idxes: list, side_idxes: list, opposite_idxes: list) -> np.array:
        idxes = [i for i in range(len(invalid_idxes)) if invalid_idxes[i]]
        if idxes:
            fix = data[np.array(idxes)[:, None], np.array(opposite_idxes)]
            data[np.array(idxes)[:, None], np.array(side_idxes)] = fix
        return data

    def __fix_missing_eyes(self, data: np.array) -> pd.DataFrame:
        """
        Copies values from valid columns to invalid ones in the same row.
        If values are missing for one eye (ValidityX = 4) but present for the other
        (ValidityX = 0) in a given row, the values associated with the valid eye are
        copied to the corresponding rows for the invalid eye.
        :param data: item whose missing eye values will be fixed
        :return: the input with eye values fixed where possible
        """

        # Get idxes of rows with invalid left but valid right eye
        invalid_left = (data[:, self.__validity_left_idx] == 4.0) & (data[:, self.__validity_right_idx] == 0.0)
        data = self.__fix_eye_side(data, invalid_left, self.__left_idxes, self.__right_idxes)

        # Get idxes of rows with invalid right but valid left eye
        invalid_right = (data[:, self.__validity_left_idx] == 0.0) & (data[:, self.__validity_right_idx] == 4.0)
        data = self.__fix_eye_side(data, invalid_right, self.__right_idxes, self.__left_idxes)

        return data

    @staticmethod
    def __replace_invalid_side(data: np.array, validity_side_idx: int) -> np.array:
        invalid_rows = (data[:, validity_side_idx] == 4.0)
        invalid_idxes = [i for i in range(len(invalid_rows)) if invalid_rows[i]]
        if invalid_idxes:
            data[np.array(invalid_idxes)[:, None], :] = np.NaN
        return data

    def __fix_invalid_rows(self, data: np.array) -> pd.DataFrame:
        """
        Replace values corresponding to an invalid eye with -1 everywhere
        :param data: item whose invalid rows will be fixed
        :return: the input with eye values fixed where possible
        """
        data = self.__replace_invalid_side(data, self.__validity_left_idx)
        data = self.__replace_invalid_side(data, self.__validity_right_idx)
        return data

    def __cyclic_split(self, sequence: pd.DataFrame, path_to_preprocessed_augmented: str, item_id: str):
        """
        Performs a cyclic split of the sequences producing augmentations
        :param sequence: the sequence to be cyclically split
        :param path_to_preprocessed_augmented: the path where to save tha augmented data at
        :param item_id: the id of the item being processed
        """
        if self.__split_step == -1:
            sequence_length = sequence.shape[0]
            if sequence_length > self.__max_sequence_length:
                self.__split_step = (sequence_length // self.__max_sequence_length) + 1
            else:
                self.__split_step = 2

        for i in range(0, self.__split_step):
            file_name = item_id + "-" + str(i + 1) + ".pkl"
            sampled_sequence = sequence[i::self.__split_step]
            sampled_sequence.to_pickle(os.path.join(path_to_preprocessed_augmented, file_name))

    def __adjust(self, item: np.array) -> np.array:
        """
        Adjusts gaze coordinates according to validity scores
        :param item: a data item made up of sequences
        :return: the input data item with adjusted gaze coordinates
        """
        num_invalid_rows = self.__percentage_partially_invalid_rows(item)
        if self.__verbose:
            print("\n Time steps with at least one invalid eye: {}%".format(num_invalid_rows))
        if num_invalid_rows > 0:
            item = self.__fix_missing_eyes(item)

        num_invalid_rows = self.__percentage_partially_invalid_rows(item)
        if self.__verbose:
            print("\n Time steps with both eyes invalid: {}% \n".format(num_invalid_rows))
        if num_invalid_rows > 0:
            item = self.__fix_invalid_rows(item)

        return pd.DataFrame(item, columns=self.__selected_features).dropna()

    def __preprocess(self, path_to_raw: str, path_to_preprocessed: str, path_to_preprocessed_augmented: str):
        file_names = [patient for patient in os.listdir(path_to_raw)]
        for file_name in tqdm(file_names, desc="Preprocessing files at {}".format(path_to_raw)):
            item_id = file_name.split('.')[0]
            sequences = pd.read_csv(os.path.join(path_to_raw, file_name))[self.__selected_features]

            if self.__verbose:
                print("\n Patient {} - Data has shape: {}".format(item_id, sequences.shape))

            if sequences.shape[0] == 0:
                if self.__verbose:
                    print("\n WARNING: Skipping patient {} since data is not available".format(item_id))
                continue

            if self.__fill_na_with_mean:
                sequences = sequences.fillna(sequences.mean())
            else:
                sequences = sequences.dropna()

            if self.__filter_by_event:
                sequences = sequences[sequences[self.__features["event"][0]] == self.__event_type.capitalize()]

            if sequences.shape[0] == 0:
                if self.__verbose:
                    print("\n WARNING: Skipping patient {} since all time steps have been filtered".format(item_id))
                continue

            sequences = self.__adjust(sequences.values)

            if self.__collapse_fixations:
                sequences = sequences.groupby(self.__features["fixations"]).size()
                sequences = sequences.reset_index().rename(columns={0: 'FixationLength'})
            else:
                # Drop useless features
                sequences = sequences.drop(labels=self.__features["event"] + self.__features["validity"], axis=1)

            # Save base sequences
            sequences.to_pickle(os.path.join(path_to_preprocessed, item_id + ".pkl"))

            # Save augmented sequences
            if self.__use_cyclic_split:
                self.__cyclic_split(sequences, path_to_preprocessed_augmented, item_id)

    def run(self):
        self.__preprocess(self.__paths_to_raw_sequences["pos"],
                          self.__paths_to_base_sequences["pos"],
                          self.__paths_to_augmented_sequences["pos"])

        self.__preprocess(self.__paths_to_raw_sequences["neg"],
                          self.__paths_to_base_sequences["neg"],
                          self.__paths_to_augmented_sequences["neg"])
