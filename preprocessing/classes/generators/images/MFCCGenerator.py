import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing.classes.base.Generator import Generator


class MFCCGenerator(Generator):

    def __init__(self):
        super().__init__("mfcc")
        self.__paths_to_audio = self._paths.create_paths(self._params["path_to_src"])
        self.__paths_to_mfcc = self._paths.create_paths(self._params["path_to_dest"])

    @staticmethod
    def __generate(path_to_src: str, path_to_destination: str):
        audio_files = os.listdir(path_to_src)
        for file_name in tqdm(audio_files, desc="Generating MFCC at {}".format(path_to_destination)):
            path_to_file = os.path.join(path_to_src, file_name)
            audio, sample_rate = librosa.load(path_to_file, res_type='kaiser_fast')
            librosa.display.specshow(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40))
            plt.savefig(os.path.join(path_to_destination, file_name.split(".")[0] + ".png"), bbox_inches='tight')
            plt.clf()

    def run(self):
        self.__generate(self.__paths_to_audio["pos"], self.__paths_to_mfcc["pos"])
        self.__generate(self.__paths_to_audio["neg"], self.__paths_to_mfcc["neg"])
