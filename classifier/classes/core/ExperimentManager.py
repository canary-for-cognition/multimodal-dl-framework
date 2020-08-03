import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch

from classifier.classes.core.CrossValidator import CrossValidator
from classifier.classes.data.Dataset import Dataset
from classifier.classes.data.splitter.DataSplitManager import DataSplitManager
from classifier.classes.utils.Params import Params


class ExperimentManager:

    def __init__(self,
                 experiment_id: str,
                 network_type: str,
                 dataset_type: str,
                 dataset: Dataset,
                 cv_metadata: list,
                 training_params: dict,
                 device: torch.device):
        """
        :param experiment_id: the id of the current experiment whose results folder will be named after
        :param network_type: the type of network to be used for the experiment
        :param dataset_type: the type of dataset to be used for the experiment
        :param dataset: an instance of the selected dataset type
        :param cv_metadata: the info about the split items for the CV
        :param training_params: the parameters concerning the training procedure
        :param device: the device on which to run the experiment
        """
        cv_params = Params.load_cv_params()

        validation_type = cv_params["type"]
        plot_metrics = cv_params["plot_metrics"]

        self.__reload_stored_split = cv_params["reload_split"]["folds"]
        self.__generate_split_from_file = cv_params["use_cv_metadata"]
        self.__save_split_metadata = cv_params["save_split_metadata"]
        self.__fold_on = cv_params["folds_type"]

        self.__data_split_manager = DataSplitManager(cv_params, dataset)
        self.__cv_splits_files = cv_metadata
        self.__device = device
        self.__cross_validator = CrossValidator(self.__data_split_manager,
                                                experiment_id,
                                                validation_type,
                                                dataset_type,
                                                network_type,
                                                training_params,
                                                plot_metrics,
                                                device)

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
        print(df)

        file_name = "aggregated_test_results_seeds.csv"
        df.to_csv(os.path.join(self.__cross_validator.get_path_to_experiment(), file_name), index=False)

        return df

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
        path_to_seed = os.path.join(self.__cross_validator.get_path_to_experiment(), "seed_" + str(seed))
        df.to_csv(os.path.join(path_to_seed, "aggregated_test_results.csv"), index=False)

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
            self.__data_split_manager.split_from_file(cv_split_file)
            avg_test_scores += [self.__cross_validator.validate(seed, iteration + 1)]
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

            if self.__reload_stored_split and self.__data_split_manager.check_split_availability():
                print("\n Reloading stored splits... \n")
                test_scores = [self.__cross_validator.validate(seed + 1)]
            else:
                if self.__generate_split_from_file:
                    print("\n Generating splits from metadata... \n")
                    test_scores = self.__run_cv_iterations(seed + 1)
                else:
                    print("\n Generating new splits... \n")
                    self.__data_split_manager.split(self.__fold_on, self.__save_split_metadata)
                    test_scores = [self.__cross_validator.validate(seed + 1)]

            aggregated_test_scores = self.__aggregate_iterations_results(avg_test_scores=test_scores,
                                                                         seed=seed + 1,
                                                                         num_iterations=len(self.__cv_splits_files))

            avg_test_scores += [aggregated_test_scores]

            print("\n................................................................\n"
                  "|                    Finished cross validation                  |\n"
                  "................................................................\n")

        self.__aggregate_seeds_results(avg_test_scores, num_seeds)

        print("\n\n==========================================================\n"
              "            Finished experiment in {:.2f}m              \n"
              "==========================================================\n".format((time.time() - start_time) / 60))
