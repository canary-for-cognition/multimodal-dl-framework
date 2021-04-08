import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.classes.base.Preprocessor import Preprocessor
from preprocessing.classes.utils.AssetLoader import AssetLoader


class ETSeqPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__("eye_tracking_sequences")

        self.__validity_left_idx, self.__validity_right_idx = None, None
        self.__left_idxes, self.__right_idxes = None, None

        self.__features = AssetLoader.load_features("eye_tracking_sequences")
        self.__selected_features = self.__select_features(self._params["features"])

        self.__verbose = self._params["verbose"]
        self.__max_seq_len = self._params["max_sequence_length"]

        self.__filter_by_event = self._params["filter_by_event"]["active"]
        if self.__filter_by_event:
            self.__event_type = self._params["filter_by_event"]["type"]

        self.__use_cyclic_split = self._params["cyclic_split"]["active"]
        if self.__use_cyclic_split:
            self.__split_step = self._params["cyclic_split"]["step"]

        self.__collapse_fixations = self._params["collapse_fixations"]

        self.__paths_to_raw_seq = self._paths.create_paths(self._params["path_to_src"])
        path_to_dest = self._params["path_to_dest"]
        self.__paths_to_base_seq = self._paths.create_paths(os.path.join(path_to_dest, "base"))
        self.__paths_to_augmented_seq = self._paths.create_paths(os.path.join(path_to_dest, "augmented"))

    def __update_idxes(self, features: List):
        self.__validity_left_idx = features.index("ValidityLeft")
        self.__validity_right_idx = features.index("ValidityRight")
        self.__left_idxes, self.__right_idxes = [], []
        for i, f in enumerate(features):
            if "Left" in f:
                self.__left_idxes.append(i)
            elif "Right" in f:
                self.__right_idxes.append(i)

    def __select_features(self, selected_features: str) -> List:
        if selected_features == "all":
            features = [f for group in self.__features.keys() for f in self.__features[group]]
        else:
            features = [f for group in selected_features.split("_") for f in self.__features[group]]
            features += self.__features["event"] + self.__features["validity"]
        self.__update_idxes(features)
        return features

    def __percentage_partially_invalid_rows(self, data: np.ndarray) -> float:
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
    def __fix_eye_side(data: np.ndarray, invalid_idxes: List, side_idxes: List, opposite_idxes: List) -> np.ndarray:
        idxes = [i for i in range(len(invalid_idxes)) if invalid_idxes[i]]
        if idxes:
            fix = data[np.array(idxes)[:, None], np.array(opposite_idxes)]
            data[np.array(idxes)[:, None], np.array(side_idxes)] = fix
        return data

    def __fix_missing_eyes(self, data: np.ndarray) -> np.ndarray:
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
    def __replace_invalid_side(data: np.ndarray, validity_side_idx: int) -> np.ndarray:
        invalid_rows = (data[:, validity_side_idx] == 4.0)
        invalid_idxes = [i for i in range(len(invalid_rows)) if invalid_rows[i]]
        if invalid_idxes:
            data[np.array(invalid_idxes)[:, None], :] = np.NaN
        return data

    def __adjust(self, seq: np.ndarray) -> pd.DataFrame:
        """ Adjusts gaze coordinates according to validity scores """

        # Fix rows with one invalid eye
        num_invalid_rows = self.__percentage_partially_invalid_rows(seq)
        if self.__verbose:
            print("\n Time steps with at least one invalid eye: {}%".format(num_invalid_rows))
        if num_invalid_rows > 0:
            seq = self.__fix_missing_eyes(seq)

        # Fix rows with both invalid eyes
        num_invalid_rows = self.__percentage_partially_invalid_rows(seq)
        if self.__verbose:
            print("\n Time steps with both eyes invalid: {}% \n".format(num_invalid_rows))
        if num_invalid_rows > 0:
            seq = self.__replace_invalid_side(seq, self.__validity_left_idx)
            seq = self.__replace_invalid_side(seq, self.__validity_right_idx)

        seq = pd.DataFrame(seq, columns=self.__selected_features).fillna(-1)
        seq = seq.drop(labels=self.__features["validity"], axis=1)

        return seq

    def __cyclic_split(self, seq: pd.DataFrame, path_to_dest_augmented: str, item_id: str):
        """
        Performs a cyclic split of the sequences producing augmentations
        :param seq: the sequence to be cyclically split
        :param path_to_dest_augmented: the path where to save tha augmented data at
        :param item_id: the id of the item being processed
        """
        # Compute split step based on desired sequence length
        if self.__split_step == -1:
            seq_len = seq.shape[0]
            self.__split_step = 2 if seq_len < self.__max_seq_len else (seq_len // self.__max_seq_len) + 1

        # Generate and save the augmented sequences
        for i in range(0, self.__split_step):
            item_id += "-" + str(i + 1)
            sampled_sequence = seq[i::self.__split_step]
            self.__save_seq(sampled_sequence, path_to_dest_augmented, item_id)

    def __save_seq(self, seq: pd.DataFrame, path_to_dest: str, item_id: str):

        # Collapse all fixations in one time step and save their length as a new feature
        if self.__collapse_fixations:
            seq["FixationLength"] = seq.groupby("FixationIndex")["FixationIndex"].transform('count')
            seq["FixationLength"][seq["FixationIndex"] == -1] = -1
            seq = seq.loc[(seq["FixationIndex"] == -1) | ~seq["FixationIndex"].duplicated()]

        # Drop useless features
        seq = seq.drop(labels=self.__features["event"], axis=1).astype(float)

        # Save base sequences
        seq.to_pickle(os.path.join(path_to_dest, item_id + ".pkl"))

    def __preprocess(self, path_to_raw: str, path_to_dest: str, path_to_dest_augmented: str):
        file_names = [patient for patient in os.listdir(path_to_raw)]
        for file_name in tqdm(file_names, desc="Preprocessing files at {}".format(path_to_raw)):
            item_id = file_name.split('.')[0]
            seq = pd.read_csv(os.path.join(path_to_raw, file_name))[self.__selected_features]

            if self.__verbose:
                print("\n Patient {} - Data has shape: {}".format(item_id, seq.shape))

            if seq.shape[0] == 0:
                if self.__verbose:
                    print("\n WARNING: Skipping patient {} since data is not available".format(item_id))
                continue

            if self.__filter_by_event:
                seq = seq[seq[self.__features["event"][0]] == self.__event_type.capitalize()]

            if seq.shape[0] == 0:
                if self.__verbose:
                    print("\n WARNING: Skipping patient {} since all time steps have been filtered".format(item_id))
                continue

            # Fix missing data
            seq = self.__adjust(seq.values)

            # Save base sequence
            self.__save_seq(seq, path_to_dest, item_id)

            # Save augmented sequence
            if self.__use_cyclic_split:
                self.__cyclic_split(seq, path_to_dest_augmented, item_id)

    def run(self):
        self.__preprocess(self.__paths_to_raw_seq["pos"],
                          self.__paths_to_base_seq["pos"],
                          self.__paths_to_augmented_seq["pos"])

        self.__preprocess(self.__paths_to_raw_seq["neg"],
                          self.__paths_to_base_seq["neg"],
                          self.__paths_to_augmented_seq["neg"])
