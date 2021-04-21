import os
import pickle

import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from preprocessing.classes.base.Generator import Generator


class ScanPathsGenerator(Generator):

    def __init__(self):
        super().__init__("scanpaths")
        self.__use_temp_grad = self._params["temporal_gradient"]
        self.__paths_to_sequences = self._paths.create_paths(self._params["path_to_src"])
        self.__paths_to_scan_paths = self._paths.create_paths(self._params["path_to_dest"])

    def __generate(self, path_to_src: str, path_to_destination: str):
        sequences_files = os.listdir(path_to_src)
        for file_name in tqdm(sequences_files, desc="Generating scan-paths at {}".format(path_to_destination)):
            item = pickle.load(open(os.path.join(path_to_src, file_name), "rb")).values
            item = item[item[:, 0] != -1.0]
            x, y = item[:, 4], item[:, 5]

            if self.__use_temp_grad:
                _, _ = plt.subplots()
                self.__color_lines(x, y)
                plt.scatter(x, y, c="k", s=1, vmin=0, vmax=1050, alpha=0.0)
            else:
                plt.scatter(x, y, vmin=0, vmax=1050)
                plt.plot(x, y)

            plt.axis('off')
            plt.savefig(os.path.join(path_to_destination, file_name.replace("pkl", "png")), bbox_inches='tight')
            plt.clf()

    @staticmethod
    def __color_lines(x: np.ndarray, y: np.ndarray):
        path = mpath.Path(np.column_stack([x, y]))
        vertices = path.interpolated(steps=3).vertices
        x, y = vertices[:, 0], vertices[:, 1]
        z = np.asarray(np.linspace(0.0, 1.0, len(x)))

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = mcoll.LineCollection(segments, array=z, cmap=plt.get_cmap('winter'),
                                  norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=0.5)
        ax = plt.gca()
        ax.add_collection(lc)

    def run(self):
        self.__generate(self.__paths_to_sequences["pos"], self.__paths_to_scan_paths["pos"])
        self.__generate(self.__paths_to_sequences["neg"], self.__paths_to_scan_paths["neg"])
