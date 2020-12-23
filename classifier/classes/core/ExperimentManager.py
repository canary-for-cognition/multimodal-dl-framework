import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch

from classifier.classes.core.CrossValidator import CrossValidator
from classifier.classes.data.Dataset import Dataset
from classifier.classes.data.DataManager import DataManager
from classifier.classes.utils.Params import Params


class ExperimentManager:

    def __init__(self, dataset: Dataset, cv_metadata: list, train_params: dict):
        """
        :param dataset: an instance of the selected dataset type
        :param cv_metadata: the info about the split items for the CV
        :param train_params: the parameters concerning the train procedure
        """
        cv_params = Params.load_cv_params()

        self.__reload_stored_split = cv_params["reload_folds"]
        self.__generate_splits_from_file = cv_params["use_cv_metadata"]

        self.__data_manager = DataManager(cv_params, dataset)
        self.__cv_splits_files = cv_metadata
        self.__device = train_params["device"]

        network_type = train_params["network_type"]
        dataset_type = dataset.get_dataset_type()

        experiment_id = "{}_{}_{}".format(dataset_type, network_type, str(time.time()))
        self.__path_to_results = os.path.join("results", experiment_id)
        os.makedirs(self.__path_to_results)
        Params.save_experiment_params(self.__path_to_results, network_type, dataset.get_dataset_type())

        self.__cv = CrossValidator(self.__data_manager, self.__path_to_results, train_params)

    @staticmethod
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

    @staticmethod
    def set_random_seed(seed: int, device: torch.device):
        """
        Sets the random seed for Numpy, Torch and python.random
        :param seed: the manual seed to be set
        :param device: the device on which to run the experiment
        """
        torch.manual_seed(seed)
        if device.type == 'cuda:3':
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __aggregate_seeds_results(self, avg_test_scores: list, num_seeds: int):
        """
        Aggregates the test results of each seed in to a report file called "aggregated_test_results_seeds.csv"
        :param avg_test_scores: the average test scores for each seed
        :param num_seeds: the number of random seeds of the experiment
        """
        print("\n Aggregated results for {} seeds \n".format(num_seeds))
        dfs = [pd.concat([pd.DataFrame({"Seed": list(range(1, s.shape[0] + 1))}), s], axis=1) for s in avg_test_scores]
        df = pd.concat(dfs)
        df.to_csv(os.path.join(self.__path_to_results, "avg_seeds_test.csv"), index=False)
        print(df)

    def __aggregate_iterations_results(self, avg_test_scores: list, seed: int, num_iterations: int) -> pd.DataFrame:
        """
        Aggregates the test results of each iteration in a report file called "aggregated_test_results.csv"
        :param avg_test_scores: the average test scores for each iteration 
        :param seed: the current random seed of the experiment
        :param num_iterations: the total number of CV iterations that have been performed
        :return the aggregated test results for the iterations
        """
        columns = avg_test_scores[0].keys()
        df = pd.DataFrame(avg_test_scores, columns=columns)
        print("\n Aggregated test results for seed # {} \n".format(seed))
        print(df.head(num_iterations))

        df = df.rename(columns={column: column.capitalize() for column in list(columns)})
        df.insert(loc=0, column="Iteration", value=range(1, num_iterations + 1))
        path_to_seed = os.path.join(self.__path_to_results, "seed_" + str(seed))
        df.to_csv(os.path.join(path_to_seed, "avg_iterations_test.csv"), index=False)

        return df

    def __run_cv_iterations(self, seed: int) -> list:
        """
        Runs one iteration of CV for each selected metadata file 
        :param seed: the current random seed of the experiment
        :return the average test results of each iteration
        """
        avg_test_scores = []
        for iteration, cv_split_file in enumerate(self.__cv_splits_files):
            print("\n\n..........................................................\n"
                  "                  CV iteration {} / {}                     \n"
                  "..........................................................\n"
                  .format(iteration + 1, len(self.__cv_splits_files)))
            self.__data_manager.split_from_file(cv_split_file)
            avg_test_scores += [self.__cv.validate(seed, iteration + 1)]
        return avg_test_scores

    def run_experiment(self, base_seed: int, num_seeds: int):
        """
        Performs the experiment running the CV for the selected number of seeds
        :param base_seed: the initial manual seed to be set (and incremented)
        :param num_seeds: the number of seeds to be used for the experiment
        """
        avg_test_scores = []
        start_time = time.time()

        for seed in range(num_seeds):
            print("\n\n==========================================================\n"
                  "                      Seed {} / {}                       \n"
                  "==========================================================\n".format(seed + 1, num_seeds))

            self.set_random_seed(base_seed + seed, self.__device)

            if self.__reload_stored_split and self.__data_manager.split_available():
                print("Reloading stored splits...")
                test_scores = [self.__cv.validate(seed + 1)]
            else:
                if self.__generate_splits_from_file:
                    print("Generating splits from metadata...")
                    test_scores = self.__run_cv_iterations(seed + 1)
                else:
                    print("Generating new splits...")
                    self.__data_manager.split()
                    test_scores = [self.__cv.validate(seed + 1)]

            avg_test_scores += [self.__aggregate_iterations_results(test_scores, seed + 1, len(self.__cv_splits_files))]

            print("\n................................................................\n"
                  "|                    Finished cross val                  |\n"
                  "................................................................\n")

        self.__aggregate_seeds_results(avg_test_scores, num_seeds)

        print("\n\n==========================================================\n"
              "            Finished experiment in {:.2f}m              \n"
              "==========================================================\n".format((time.time() - start_time) / 60))
