import os
import shutil

import pandas as pd
from tqdm import tqdm


def get_preprocessed_file_name(filenames_to_pid_map: Dict, task: str, raw_filename: str) -> str:
    """
    Creates and returns the PID-based file name for the preprocessed data item
    :param filenames_to_pid_map: a map from raw file names to PIDs
    :param task: the name of the current task
    :param raw_filename: the raw file name of the currently processed item
    :return: the PID-based file name for the preprocessed data item
    """
    pid = filenames_to_pid_map[filenames_to_pid_map[task] == raw_filename[:-4]]["pid"].values[0]
    return pid + ".png"


def get_paths_to_preprocessed(path_to_task: str, labels_map: Dict) -> Dict:
    """
    Creates and returns the paths to the preprocessed data items
    :param path_to_task: the path to the data related to the currently processed task
    :param labels_map: a map with the name of positive and negative labels
    :return:
    """
    base_path_to_preprocessed = os.path.join(path_to_task, "preprocessed")
    path_to_preprocessed_pos = os.path.join(base_path_to_preprocessed, labels_map["pos"])
    path_to_preprocessed_neg = os.path.join(base_path_to_preprocessed, labels_map["neg"])
    os.makedirs(base_path_to_preprocessed, exist_ok=True)
    os.makedirs(path_to_preprocessed_pos, exist_ok=True)
    os.makedirs(path_to_preprocessed_neg, exist_ok=True)
    return {
        "neg": path_to_preprocessed_neg,
        "pos": path_to_preprocessed_pos
    }


def main():
    # Path to the file containing the CSV mapping filenames to PIDs for each task
    path_to_filenames_to_pid_map = os.path.join("metadata", "heatmaps_name_to_pid.csv")

    # Labels for the classification problem
    negative_label = "0_healthy"
    positive_label = "1_alzheimer"

    # List of tasks matching the names of the subdirectories in "./tasks"
    tasks = ["cookie_theft", "memory", "reading"]

    filenames_to_pid_map = pd.read_csv(path_to_filenames_to_pid_map)

    labels_map = {
        "neg": negative_label,
        "pos": positive_label
    }

    # Iterate over tasks
    for task in tasks:
        path_to_task = os.path.join("tasks", task)
        path_to_raw = os.path.join(path_to_task, "raw")
        paths_to_preprocessed = get_paths_to_preprocessed(path_to_task, labels_map)

        # Iterate over files for each task
        for raw_filename in tqdm(os.listdir(path_to_raw), desc="\n Processing files for task: {} \n".format(task)):
            preprocessed_filename = get_preprocessed_file_name(filenames_to_pid_map, task, raw_filename)
            label_id = preprocessed_filename[0]
            path_to_preprocessed_file = os.path.join(paths_to_preprocessed[label_id], preprocessed_filename)
            path_to_raw_file = os.path.join(path_to_raw, raw_filename)
            shutil.copy(path_to_raw_file, path_to_preprocessed_file)


if __name__ == '__main__':
    main()
