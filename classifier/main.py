import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch

from classifier.classes.core.CrossValidator import CrossValidator
from classifier.classes.data.DataManager import DataManager
from classifier.classes.utils.Params import Params


def set_random_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == 'cuda:3':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(device_type: str) -> torch.device:
    """
    Returns the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    :param device_type: the id of the selected device (if cuda device, must match the regex "cuda:\d"
    :return: the device specified in the experiments parameters (if available, else fallback to a "cpu" device")
    """
    if device_type == "cpu":
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", device_type):
        if not torch.cuda.is_available():
            print("WARNING: running on cpu since device {} is not available".format(device_type))
            return torch.device("cpu")
        return torch.device(device_type)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(device_type))


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_params, data_params, num_seeds, device_type = Params.load()

    device = get_device(device_type)
    set_random_seed(0, device)
    train_params["device"] = device

    network_type = train_params["network_type"]
    dataset_name = data_params["dataset"]["name"]

    print("\n\n==========================================================\n"
          "            Experiment on {} using {}                       \n"
          "==========================================================\n\n".format(dataset_name, network_type))

    print("\t Using Torch version ... : {}".format(torch.__version__))
    print("\t Running on device ..... : {}\n".format(device))

    experiment_id = "{}_{}_{}".format(data_params["dataset"]["name"], network_type, str(time.time()))
    path_to_results = os.path.join("results", experiment_id)
    os.makedirs(path_to_results)
    Params.save_experiment_params(path_to_results, network_type, dataset_name)

    test_scores = []
    start_time = time.time()

    for seed in range(num_seeds):
        print("\n\n==========================================================\n"
              "                      Seed {} / {}                       \n"
              "==========================================================\n".format(seed + 1, num_seeds))

        data_manager = DataManager(data_params, network_type)
        use_cv_metadata = data_params["cv"]["use_cv_metadata"]

        if use_cv_metadata:
            path_to_metadata = data_params["dataset"]["paths"]["cv_metadata"]
            data_manager.reload_split(path_to_metadata, seed + 1)
        else:
            data_manager.generate_split()
            data_manager.save_split_to_file(path_to_results, seed + 1)

        data_manager.print_split_info()

        cv = CrossValidator(data_manager, path_to_results, train_params)

        set_random_seed(seed, device)
        test_scores += [cv.validate(seed + 1)]

        print("\n................................................................\n"
              "                        Finished CV                  \n"
              "................................................................\n")

    print("\n Aggregated results for {} seeds \n".format(num_seeds))
    dfs = [pd.concat([pd.DataFrame({"Seed": list(range(1, s.shape[0] + 1))}), s], axis=1) for s in test_scores]
    df = pd.concat(dfs)
    df.to_csv(os.path.join(path_to_results, "avg_seeds_test.csv"), index=False)
    print(df)

    print("\n\n==========================================================\n"
          "            Finished experiment in {:.2f}m              \n"
          "==========================================================\n".format((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()
