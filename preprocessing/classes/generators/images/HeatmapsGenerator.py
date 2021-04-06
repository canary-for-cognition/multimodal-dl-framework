import os
import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from preprocessing.classes.base.Generator import Generator


class HeatmapsGenerator(Generator):

    def __init__(self):
        super().__init__("heatmaps")

        self.__debug_mode = self._params["debug_mode"]
        self.__filter_out_non_fixations = self._params["filter_out_non_fixations"]
        self.__max_sequence_length = self._params["max_sequence_length"]

        self.__show_bg_image = self._params["bg_image"]["show"]
        if self.__show_bg_image:
            self.__path_to_background_img = self._paths.get_metadata(self._params["bg_image"]["filename"])

        self.__use_proportions = self._params["proportions"]["active"]
        if self.__use_proportions:
            self.__origin = self._params["proportions"]["origin"]
            self.__width = self._params["proportions"]["width"]
            self.__height = self._params["proportions"]["height"]

        self.__dataset_type = self._paths.get_dataset_type()
        self.__paths_to_sequences = self._paths.create_paths(self._params["path_to_src"])
        self.__paths_to_heatmaps = self._paths.create_paths(self._params["path_to_dest"])

    @staticmethod
    def __data_coord2view_coord(p, resolution, p_min, p_max) -> float:
        return (p - p_min) / (p_max - p_min) * resolution

    @staticmethod
    def __nearest_neighbour_density(xv, yv, resolution, neighbours, dim=2) -> np.ndarray:
        # Find the closest nn_max-1 neighbors (first entry is the point itself)
        grid = np.mgrid[0:resolution, 0:resolution].T.reshape(resolution ** 2, dim)
        dists = cKDTree(np.array([xv, yv]).T).query(grid, neighbours)

        # Inverse of the sum of distances to each grid point
        inv_sum_dists = 1. / dists[0].sum(1)

        return inv_sum_dists.reshape(resolution, resolution)

    @staticmethod
    def __get_transparent_colors():
        color_map = plt.cm.get_cmap("jet")
        color_map._init()
        color_map._lut[:, -1] = np.linspace(0, 1, 259)
        return color_map

    # @staticmethod
    # def __kde_quartic(d, h):
    #     """ Computes the heat intensity with quartic kernel """
    #     dn = d / h
    #     return (15 / 16) * (1 - dn ** 2) ** 2
    #
    # def __generate_heatmap_with_proportions(self, x: np.ndarray, y: np.ndarray, path_to_heatmap: str):
    #     x = list(x.astype(int))
    #     y = list(y.astype(int))
    #
    #     grid_size = 10
    #     h = 100
    #
    #     x_min, x_max = 0, self.__width
    #     y_min, y_max = 0, self.__height
    #
    #     # Construct the grid
    #     x_grid = np.arange(x_min - h, x_max + h, grid_size)
    #     y_grid = np.arange(y_min - h, y_max + h, grid_size)
    #     x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    #
    #     # Get the center point of the grid
    #     xc = x_mesh + (grid_size / 2)
    #     yc = y_mesh + (grid_size / 2)
    #
    #     intensity_list = []
    #     for j in range(len(xc)):
    #         intensity_row = []
    #         for k in range(len(xc[0])):
    #             kde_value_list = []
    #             for i in range(len(x)):
    #                 d = math.sqrt((xc[j][k] - x[i]) ** 2 + (yc[j][k] - y[i]) ** 2)
    #                 kde_value_list.append(self.__kde_quartic(d, h) if d <= h else 0)
    #             p_total = sum(kde_value_list)
    #             intensity_row.append(p_total)
    #         intensity_list.append(intensity_row)
    #
    #     intensity = np.array(intensity_list)
    #     plt.axis('off')
    #     plt.pcolormesh(x_mesh, y_mesh, intensity)
    #
    #     if self.__debug_mode:
    #         # plt.plot(x, y, 'ro', alpha=0.1)
    #         # plt.colorbar()
    #         plt.title(path_to_heatmap.split(os.sep)[-1])
    #         plt.show()
    #         exit("debug image proportions heatmap")
    #
    #     plt.savefig(path_to_heatmap, bbox_inches='tight')

    def __generate_heatmap_with_proportions(self, x: np.ndarray, y: np.ndarray, path_to_heatmap: str):
        extent = [0, self.__width, 0, self.__height]
        fig, ax = plt.subplots()

        if self.__show_bg_image:
            ax.imshow(Image.open(self.__path_to_background_img), origin="upper", extent=extent)

        heat_map, _, _ = np.histogram2d(x, y, bins=100, range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        heat_map = gaussian_filter(heat_map, sigma=2)
        ax.imshow(heat_map.T, origin=self.__origin, extent=extent, cmap=self.__get_transparent_colors())
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.axis('off')

        if self.__debug_mode:
            # plt.plot(x, y, 'ro', alpha=0.1)
            # plt.colorbar()
            plt.title(path_to_heatmap.split(os.sep)[-1])
            plt.show()
            exit("debug image proportions heatmap")

        plt.savefig(path_to_heatmap, bbox_inches='tight')
        plt.clf()

    def __generate_heatmap(self, x: np.ndarray, y: np.ndarray, path_to_heatmap: str):
        fig, ax = plt.subplots()

        heat_map, _, _ = np.histogram2d(x, y, bins=100)
        heat_map = gaussian_filter(heat_map, sigma=2)

        ax.imshow(heat_map.T, cmap=self.__get_transparent_colors())
        ax.axis('off')
        if self.__debug_mode:
            plt.show()
            exit("debug self proportions heatmap")

        plt.savefig(path_to_heatmap, bbox_inches='tight')
        plt.clf()

    @staticmethod
    def __fetch_confusion_coords(item: np.ndarray, file_name: str) -> Tuple:
        # Column 0 is now avg Gx
        item[:, 0] = (item[:, 0] + item[:, 7]) / 2

        # Column 1 is now avg Gy
        item[:, 1] = (item[:, 1] + item[:, 8]) / 2

        item = item[item[:, 0] > 0]
        item = abs(item)
        item[item == 1.0] = -1.0

        # Determine layout (horizontal or vertical) for task
        orientation = file_name.split('-')[0][-1]

        if orientation == 'H':
            item = item[
                (item[:, 0] < 775.0) | (item[:, 0] > 880.0) | (item[:, 1] < 334.0) | (item[:, 1] > 419.0)]
        elif orientation == 'V':
            item = item[
                (item[:, 0] < 928.0) | (item[:, 0] > 1034.0) | (item[:, 1] < 226.0) | (item[:, 1] > 312.0)]
        else:
            raise ValueError("{} is not a valid orientation!".format(orientation))

        return item[:, 0], item[:, 1]

    def __fetch_alzheimer_coords(self, item: np.ndarray) -> Tuple:
        if self.__filter_out_non_fixations:
            item = item[item[:, 18] == "Fixation"]
            # Fixations x, y coordinates
            return item[:, 16], item[:, 17]
        else:
            # Gaze x, y coordinates
            # return item[:, 4], item[:, 5]
            return item[:, 0], item[:, 1]

    def __generate(self, path_to_src: str, path_to_destination: str):
        for file_name in tqdm(os.listdir(path_to_src), desc="Generating heatmaps at {}".format(path_to_destination)):
            path_to_heatmap = os.path.join(path_to_destination, file_name.rstrip("pkl") + "png")
            item = pickle.load(open(os.path.join(path_to_src, file_name), "rb"))
            item = item.values if self.__max_sequence_length == -1 else item.values[-self.__max_sequence_length:, :]

            if self.__dataset_type == "confusion":
                x, y = self.__fetch_confusion_coords(item, file_name)
            elif self.__dataset_type == "alzheimer":
                x, y = self.__fetch_alzheimer_coords(item)
            else:
                raise ValueError("'{}' is not a supported dataset!")

            if self.__use_proportions:
                self.__generate_heatmap_with_proportions(x, y, path_to_heatmap)
            else:
                self.__generate_heatmap(x, y, path_to_heatmap)

    def run(self):
        self.__generate(self.__paths_to_sequences["pos"], self.__paths_to_heatmaps["pos"])
        self.__generate(self.__paths_to_sequences["neg"], self.__paths_to_heatmaps["neg"])
