import os
import pickle
from typing import Tuple

import matplotlib.pyplot as plt
from tqdm import tqdm


def get_avg_coords(path_to_items: str) -> Tuple:
    x, y = [], []
    for filename in tqdm(os.listdir(path_to_items), desc="Computing averages at {}".format(path_to_items)):
        item = pickle.load(open(os.path.join(path_to_items, filename), "rb")).values
        x.append(item[:, 6].mean())
        y.append(item[:, 7].mean())
    return x, y


def main():
    dataset_name = "confusion"
    data_source = "eye_tracking"
    data_type = ""
    data_dimension = "augmented"
    labels = {
        "pos": "1_confused",
        "neg": "0_not_confused",
    }

    path_to_modalities = os.path.join("", "..", "..", "dataset", dataset_name, "modalities")
    path_to_sequences = os.path.join(path_to_modalities, "sequences", data_source, data_type, data_dimension)
    paths_to_sequences = {
        "pos": os.path.join(path_to_sequences, labels["pos"]),
        "neg": os.path.join(path_to_sequences, labels["neg"])
    }

    x_pos, y_pos = get_avg_coords(paths_to_sequences["pos"])
    x_neg, y_neg = get_avg_coords(paths_to_sequences["neg"])

    x, y = x_pos + x_neg, y_pos + y_neg

    plt.scatter(x, y)
    plt.xlabel("Avg X coord")
    plt.ylabel("Avg Y coord")
    plt.show()


if __name__ == '__main__':
    main()
