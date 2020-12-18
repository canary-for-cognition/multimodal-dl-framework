import os
import time
from typing import Union

import numpy as np
import torch

from classifier.classes.core.Evaluator import Evaluator
from classifier.classes.core.Trainer import Trainer
from classifier.classes.data.splitter.DataSplitManager import DataSplitManager
from classifier.classes.utils.Params import Params
from classifier.classes.utils.Plotter import Plotter


class CrossValidator:
    def __init__(self,
                 dsm: DataSplitManager,
                 experiment_id: str,
                 validation_type: str,
                 dataset_type: str,
                 network_type: str,
                 training_params: dict,
                 plot_metrics: bool,
                 device: torch.device):
        """
        :param dsm: an instance of DataSplitManager to load the folds from the filesystem
        :param validation_type: the type of CV to be performed (in ["k_fold", "leave_one_out"])
        :param dataset_type: the type of data to be processed
        :param network_type: the type of network architecture to be used for the CV
        :param training_params: the params to be submitted to the Trainer instance
        :param plot_metrics: whether or not to plot the metrics
        :param device: the device to be used (in ["cpu", "cuda:n"])
        """
        self.__data_split_manager = dsm
        self.__training_params = training_params
        self.__network_type = network_type
        self.__dataset_type = dataset_type
        self.__validation_type = validation_type
        self.__device = device
        self.__plot_metrics = plot_metrics
        self.__evaluator = Evaluator(training_params["batch_size"], device)
        self.__seed, self.__iteration = None, None
        self.__paths_to_results = {}

        self.__path_to_experiment = self.__create_path_to_experiment(experiment_id)
        Params.save_experiment_params(self.__path_to_experiment, self.__network_type, self.__dataset_type)

    def get_path_to_experiment(self) -> str:
        return self.__path_to_experiment

    def __set_seed(self, seed: int):
        self.__seed = seed

    def __set_iteration(self, iteration: int):
        self.__iteration = iteration

    def __create_path_to_experiment(self, experiment_id: str) -> str:
        timestamp = str(time.time())
        if experiment_id:
            experiment_folder = experiment_id + "_" + timestamp
        else:
            experiment_folder = "_".join([self.__dataset_type, self.__validation_type, self.__network_type, timestamp])

        path_to_experiment = os.path.join("experiments", "results", experiment_folder)
        os.makedirs(path_to_experiment)

        return path_to_experiment

    @staticmethod
    def __merge_metrics(metrics: list, set_type: str) -> dict:
        """
        Averages the metrics by set type (in ["training", "validation", "test"])
        :param metrics: the metrics of each processed fold
        :param set_type:
        :return: set code in ["training", "validation", "test"]
        """
        return {k: np.array([m[set_type][k] for m in metrics]).mean() for k in metrics[0][set_type].keys()}

    @staticmethod
    def __fetch_best_model_metrics(best_model_evaluation: dict) -> dict:
        """
        Fetches the metrics from the dictionary containing the evaluation of the best model
        :param best_model_evaluation: a dict containing the full evaluation of the best model
        :return: a dict containing the metrics of the best model
        """
        return {
            "training": best_model_evaluation["training"]["metrics"],
            "validation": best_model_evaluation["validation"]["metrics"],
            "test": best_model_evaluation["test"]["metrics"]
        }

    @staticmethod
    def __print_metrics(metrics: dict, set_type: str):
        """
        Prints the overview on the metrics for the given set type (in ["training", "validation", "test"])
        :param metrics: the metrics related to the selected set type
        :param set_type: a set code in ["training", "validation", "test"]
        """
        print("\n Average {} metrics: \n".format(set_type))
        for metric, value in metrics.items():
            print(("\t - {} " + "".join(["."] * (15 - len(metric))) + " : {}").format(metric, value))

    @staticmethod
    def __print_time_overview(recorded_time: tuple, zero_time: float, estimated_time: float):
        """
        Prints an overview on the elapsed and estimated completion time for the CV iteration
        :param recorded_time: the (start time, end time) couple for the latest training procedure
        :param zero_time: the time at which the experiment started
        :param estimated_time: the estimated time of completion for the current CV iteration
        """
        start_time, end_time = recorded_time
        print("\n Time overview: \n"
              "\n\t - Time to train current fold ........... : {:.2f}m"
              "\n\t - Elapsed time since the CV started: ... : {:.2f}m"
              "\n\t - Estimated time of completion ......... : {:.2f}m"
              .format((end_time - start_time) / 60, (end_time - zero_time) / 60, estimated_time))

    def __print_fold_heading(self, i: int, k: int):
        """
        Prints the current fold number (out of K), seed and iteration
        :param i: the number of the current fold
        :param k: the total number of folds to be processed
        """
        fold_heading = "\n * Processing fold {} / {} - seed {} ".format(i, k, self.__seed)
        fold_heading += "* \n" if self.__iteration is None else "- iteration {} * \n".format(self.__iteration)
        print(fold_heading)

    def __avg_iteration_metrics(self, cv_metrics: list, save: bool = False, inplace: bool = False) -> Union[dict, None]:
        """
        Computes the average metrics for the current CV iteration
        :param cv_metrics: the list of metrics for each processed fold of the CV iteration
        :param save: whether or not to save to file the average metrics
        :param inplace: whether or not to return the average metrics
        """
        avg_scores = {}
        for set_type in ["training", "validation", "test"]:
            set_avg_metrics = self.__merge_metrics(cv_metrics, set_type)
            self.__print_metrics(set_avg_metrics, set_type)
            avg_scores[set_type] = set_avg_metrics

        if save:
            Params.save(avg_scores, os.path.join(self.__paths_to_results["metrics"], "cv_average.json"))

        if not inplace:
            return avg_scores

    def __overall_test_evaluation(self, test_preds: list):
        """
        Computes the overall test evaluation on all the predictions made by the model on all the different folds
        :param test_preds: a list of test predictions and corresponding ground truth for each fold
        """
        metrics = self.__evaluator.evaluate_preds(test_preds)
        self.__print_metrics(metrics, set_type="overall test")
        print("\n Saving overall test evaluation metrics... \n")
        Params.save(metrics, os.path.join(self.__paths_to_results["metrics"], "test_evaluation.json"))

    def __create_paths_to_results(self):
        """
        Creates the paths to the "metrics", "models", "predictions" and "plots" folder for the current experiment.
        If multiple iterations of CV are performed, the nesting of the directories is the following:
        + cv_type_network_type_timestamp
            + seed_n
                + iteration_n
                    - metrics
                    - models
                    - plots
                    - predictions
        Otherwise, the iteration_n folder is skipped and the sub-folders belong to the seed_n directory
        """
        path_to_main_dir = os.path.join(self.__path_to_experiment, "seed_" + str(self.__seed))

        if self.__iteration is not None:
            path_to_main_dir = os.path.join(path_to_main_dir, "iteration_" + str(self.__iteration))

        self.__paths_to_results = {
            "metrics": os.path.join(path_to_main_dir, "metrics"),
            "models": os.path.join(path_to_main_dir, "models"),
            "preds": os.path.join(path_to_main_dir, "predictions"),
            "plots": os.path.join(path_to_main_dir, "plots")
        }

        for path in self.__paths_to_results.values():
            os.makedirs(path)

    def __train_fold(self, fold_number: int) -> tuple:
        """
        Trains the model on the given CV split
        :param fold_number: the CV split to be used for the training procedure
        :return: the trained model, the training/validation/test metrics for the training procedure,
                 the evaluation of the best model found and the recorded time as a couple (start time, end time)
        """
        model_name = self.__network_type + "_fold_" + str(fold_number)
        path_to_best_model = os.path.join(self.__paths_to_results["models"], model_name + ".pth")
        trainer = Trainer(self.__network_type, self.__training_params, self.__device, path_to_best_model)

        data = self.__data_split_manager.load_split(fold_number)

        start_time = time.time()
        metrics = trainer.train(data)
        end_time = time.time()

        trained_model = trainer.get_model()
        best_model_evaluation = self.__evaluator.evaluate_model(data, trained_model, path_to_best_model)

        return trained_model, metrics, best_model_evaluation, (start_time, end_time)

    def __process_folds(self) -> tuple:
        """
        Processes the K folds of the CV training the model on each of them
        :return: the average CV metrics and the aggregated test predictions for all the processed folds
        """
        test_preds, cv_metrics, folds_times = [], [], []
        plotter = Plotter(self.__paths_to_results["plots"])
        zero_time = time.time()

        k = self.__data_split_manager.get_k()

        for i in range(1, k + 1):
            self.__print_fold_heading(i, k)

            trained_model, metrics, best_model_evaluation, recorded_time = self.__train_fold(i)
            test_preds.append(best_model_evaluation["test"]["predictions"])
            folds_times.append((recorded_time[1] - recorded_time[0]) / 60)

            print("\n *** Finished iteration {} of cross validation! *** \n".format(i))

            print("\n Saving metrics... \n")
            best_metrics = self.__fetch_best_model_metrics(best_model_evaluation)
            Params.save(best_metrics, os.path.join(self.__paths_to_results["metrics"], "fold_" + str(i) + ".json"))
            cv_metrics.append(best_metrics)

            print("\n Saving predictions... \n")
            Params.save_experiment_preds(best_model_evaluation, self.__paths_to_results["preds"], i)

            if self.__plot_metrics:
                print("\n Saving plots... \n")
                plotter.plot_metrics(metrics, i)

            print("\n\n Computing average metrics for the processed folds... \n")
            self.__avg_iteration_metrics(cv_metrics, inplace=True)

            estimated_time = (np.mean(np.array(folds_times)) * (k - i))
            self.__print_time_overview(recorded_time, zero_time, estimated_time)

            print("\n----------------------------------------------------------------\n")

        return cv_metrics, test_preds

    def validate(self, seed: int, iteration: int = None):
        """
        Performs the CV
        :param seed: the seed number of the CV
        :param iteration: the iteration number of the CV (None for single-iterations experiments)
        """
        self.__set_seed(seed)
        self.__set_iteration(iteration)
        self.__create_paths_to_results()

        print("\n Running {} cross validation... \n\n".format(self.__validation_type))
        cv_metrics, test_preds = self.__process_folds()

        if self.__validation_type == "leave_one_out":
            self.__overall_test_evaluation(test_preds)

        return self.__avg_iteration_metrics(cv_metrics, save=True)["test"]
