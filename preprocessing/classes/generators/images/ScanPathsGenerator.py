import os
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing.classes.base.Generator import Generator


class ScanPathsGenerator(Generator):

    def __init__(self):
        super().__init__("scanpaths")
        self.__paths_to_sequences = self._paths.create_paths(self._params["path_to_src"])
        self.__paths_to_scan_paths = self._paths.create_paths(self._params["path_to_dest"])

    @staticmethod
    def __generate(path_to_src: str, path_to_destination: str):
        sequences_files = os.listdir(path_to_src)
        for file_name in tqdm(sequences_files, desc="Generating scan-paths at {}".format(path_to_destination)):
            item = pickle.load(open(os.path.join(path_to_src, file_name), "rb")).values
            item = item[item[:, 0] != -1.0]
            plt.scatter(item[:, 4], item[:, 5], vmin=0, vmax=1050)
            plt.plot(item[:, 4], item[:, 5])
            plt.axis('off')
            plt.savefig(os.path.join(path_to_destination, file_name[:-3] + "png"), bbox_inches='tight')
            plt.clf()

    def run(self):
        self.__generate(self.__paths_to_sequences["pos"], self.__paths_to_scan_paths["pos"])
        self.__generate(self.__paths_to_sequences["neg"], self.__paths_to_scan_paths["neg"])
