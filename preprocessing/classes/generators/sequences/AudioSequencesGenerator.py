import os

import numpy as np
import pandas as pd
import scipy.stats.stats as st
from pydub import AudioSegment
from pydub.utils import make_chunks
from python_speech_features import mfcc
from tqdm import tqdm

from preprocessing.classes.base.Generator import Generator


class AudioSequencesGenerator(Generator):

    def __init__(self):
        super().__init__("audio_sequences")

        self.__sample_rate = self._params["sample_rate"]

        self.__paths_to_raw_sequences = self._paths.get_paths_to_modality(self._params["paths"]["source"])
        self.__paths_to_base_sequences = self._paths.get_paths_to_modality(self._params["paths"]["destination"])

    @staticmethod
    def __get_summary_stats(key: str, data: np.array, coefficient: int) -> dict:
        return {
            key + "_mean": data[:, coefficient].mean(),
            key + "_variance": data[:, coefficient].var(),
            key + "_skewness": st.skew(data[:, coefficient]),
            key + "_kurtosis": st.kurtosis(data[:, coefficient])
        }

    def __extract_features(self, mfcc_data: dict) -> dict:
        """
        Extracts the features from the MFCC data
        :param mfcc_data: MFCC data for an audio chunk
        :return: the extracted features from the input MFCC data
        """
        features, mfcc_means = {}, []

        for i in range(0, 14):
            key = "energy" if i == 0 else "mfcc_" + str(i)

            features.update(self.__get_summary_stats(key, mfcc_data["mfcc_features"], i))
            features.update(self.__get_summary_stats(key + "_velocity", mfcc_data["velocity"], i))
            features.update(self.__get_summary_stats(key + "_acceleration", mfcc_data["acceleration"], i))

            if i > 0:
                mfcc_means.append(features[key + "_mean"])

        features["mfcc_skewness"] = st.skew(np.array(mfcc_means))
        features["mfcc_kurtosis"] = st.kurtosis(mfcc_means)

        return features

    def __extract_mfcc_data(self, audio_data: np.array) -> dict:
        """
        Returns the MFCC data with respect to the input audio data
        :param audio_data: audio data extracted from a WAV file
        :return: the MFCC data with respect to the input audio data
        """
        mfcc_features = mfcc(audio_data, self.__sample_rate, numcep=15, appendEnergy=True)
        velocity = (mfcc_features[:-1, :] - mfcc_features[1:, :]) / 2.0
        acceleration = (velocity[:-1, :] - velocity[1:, :]) / 2.0

        return {
            "mfcc_features": mfcc_features,
            "velocity": velocity,
            "acceleration": acceleration
        }

    def __get_features_from_chunks(self, chunks: list) -> dict:
        """
        Generates the MFCC features for the given chunks of audio file
        :param chunks: a list of chunks of audio file
        :return: a dictionary containing mean, variance, skewness, and kurtosis of the first 14 MFCCs of each chunk
        """
        features = {}
        for chunk in chunks:
            chunk = np.frombuffer(chunk.get_array_of_samples(), dtype=np.int16)
            mfcc_data = self.__extract_mfcc_data(chunk)
            chunk_features = self.__extract_features(mfcc_data)
            features = {k: features.setdefault(k, []) + [v] for k, v in chunk_features.items()}
        return features

    def __generate(self, path_to_src: str, path_to_destination: str):
        for item in tqdm(os.listdir(path_to_src), desc="Generating features for files at {}".format(path_to_src)):
            path_to_item = os.path.join(path_to_src, item)
            audio_segment = AudioSegment.from_file(file=path_to_item, format="wav", frame_rate=self.__sample_rate)
            chunks = make_chunks(audio_segment, chunk_length=1000)
            chunks = chunks[:-1] if len(chunks[-1]) < 900 else chunks
            features = self.__get_features_from_chunks(chunks)
            pd.DataFrame(features).to_csv(os.path.join(path_to_destination, item.rstrip(".wav") + ".csv"), index=False)

    def run(self):
        self.__generate(self.__paths_to_raw_sequences["pos"], self.__paths_to_base_sequences["pos"])
        self.__generate(self.__paths_to_raw_sequences["neg"], self.__paths_to_base_sequences["neg"])
