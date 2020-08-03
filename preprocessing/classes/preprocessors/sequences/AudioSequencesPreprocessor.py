import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.classes.base.Preprocessor import Preprocessor


class AudioSequencesPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__("audio_sequences")
        self.__paths_to_raw_sequences = self._paths.get_paths_to_modality(self._params["paths"]["source"])
        self.__paths_to_base_sequences = self._paths.get_paths_to_modality(self._params["paths"]["destination"])
        self._params["paths"]["destination"]["data_dimension"] = "augmented"
        self.__paths_to_augmented_sequences = self._paths.get_paths_to_modality(self._params["paths"]["destination"])

    @staticmethod
    def __to_pkl(data: pd.DataFrame, path_to_pkl_base: str, path_to_pkl_augmented: str, pid: str):
        data.to_pickle(os.path.join(path_to_pkl_base, pid + ".pkl"))
        for i, chunk in enumerate(np.array_split(data, 4)):
            chunk.to_pickle(os.path.join(path_to_pkl_augmented, pid + "-" + str(i) + ".pkl"))

    def __preprocess(self, path_to_csv: str, path_to_pkl: str, path_to_pkl_augmented: str):
        patients = [patient for patient in os.listdir(path_to_csv)]
        for patient in tqdm(patients, desc="Preprocessing files at {}".format(path_to_csv)):
            pid = patient.split('.')[0]
            path_to_patient = os.path.join(path_to_csv, patient)
            data = pd.read_csv(path_to_patient)

            nan_columns = data.columns[data.isna().any()].tolist()
            if nan_columns:
                print("\n NaN values at patient {}: \n".format(pid))
                print(data[nan_columns], "\n\n")

            self.__to_pkl(data, path_to_pkl, path_to_pkl_augmented, pid)

    def run(self):
        self.__preprocess(self.__paths_to_raw_sequences["pos"],
                          self.__paths_to_base_sequences["pos"],
                          self.__paths_to_augmented_sequences["pos"])

        self.__preprocess(self.__paths_to_raw_sequences["neg"],
                          self.__paths_to_base_sequences["neg"],
                          self.__paths_to_augmented_sequences["neg"])
