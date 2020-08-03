import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

from preprocessing.classes.base.Generator import Generator


class HSHGenerator(Generator):

    def __init__(self):
        super().__init__("hsh")
        self.__paths_to_sequences = self._paths.get_paths_to_modality(self._params["paths"]["source"])
        self.__paths_to_hsh = self._paths.get_paths_to_modality(self._params["paths"]["destination"])

    @staticmethod
    def __generate(path_to_src: str, path_to_destination: str):
        sequences_files = os.listdir(path_to_src)
        for file_name in tqdm(sequences_files, desc="Generating HSH at {}".format(path_to_destination)):
            item = pickle.load(open(os.path.join(path_to_src, file_name), "rb")).values
            item = item[item[:, 0] != -1.0]
            x, y = item[:, 4], item[:, 5]
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            plt.scatter(x, y, c=z, s=100, edgecolor=[])
            plt.plot(x, y, color="black", alpha=0.1)

            plt.axis('off')
            plt.savefig(os.path.join(path_to_destination, file_name[:-3] + "png"), bbox_inches='tight')
            plt.clf()

    def run(self):
        self.__generate(self.__paths_to_sequences["pos"], self.__paths_to_hsh["pos"])
        self.__generate(self.__paths_to_sequences["neg"], self.__paths_to_hsh["neg"])
