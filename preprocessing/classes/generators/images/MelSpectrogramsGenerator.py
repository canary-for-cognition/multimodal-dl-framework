import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from preprocessing.classes.base.Generator import Generator


class MelSpectrogramsGenerator(Generator):

    def __init__(self):
        super().__init__("mel_spectrograms")
        self.__paths_to_audio = self._paths.get_paths_to_modality(self._params["paths"]["source"])
        self.__paths_to_mel_spectrograms = self._paths.get_paths_to_modality(self._params["paths"]["destination"])

    @staticmethod
    def __generate(path_to_src: str, path_to_destination: str):
        audio_files = os.listdir(path_to_src)
        for file_name in tqdm(audio_files, desc="Generating mel-spectrograms at {}".format(path_to_destination)):
            path_to_file = os.path.join(path_to_src, file_name)
            audio, sample_rate = librosa.load(path_to_file, res_type='kaiser_fast')

            spectrogram = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=128)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            librosa.display.specshow(log_spectrogram)

            plt.savefig(os.path.join(path_to_destination, file_name.split(".")[0] + ".png"), bbox_inches='tight')
            plt.clf()

    def run(self):
        self.__generate(self.__paths_to_audio["pos"], self.__paths_to_mel_spectrograms["pos"])
        self.__generate(self.__paths_to_audio["neg"], self.__paths_to_mel_spectrograms["neg"])
