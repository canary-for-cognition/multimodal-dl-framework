import os
import time
from typing import Union

import numpy as np

from classifier.classes.core.Evaluator import Evaluator
from classifier.classes.core.Trainer import Trainer
from classifier.classes.data.DataManager import DataManager
from classifier.classes.utils.Params import Params
from classifier.classes.utils.Plotter import Plotter


class CrossValidator:

    def __init__(self, data_manager: DataManager, path_to_results: str, train_params: dict):
        """
        :param data_manager: an instance of DataManager to load the folds from the filesystem
        :param path_to_results: the path to the directory with the results for the current experiment
        :param train_params: the params to be submitted to the Trainer instance
        """
        self.__data_manager = data_manager
        self.__path_to_results = path_to_results
        self.__train_params = train_params
        self.__network_type = train_params["network_type"]
        self.__evaluator = Evaluator(train_params["device"])
        self.__seed, self.__iteration = None, None
        self.__paths_to_results = {}

    @staticmethod
    def __merge_metrics(metrics: list, set_type: str) -> dict:
        """
        Averages the metrics by set type (in ["train", "val", "test"])
        :param metrics: the metrics of each processed fold
        :param set_type: set code in ["train", "val", "test"]
        :return: the input metrics averaged by set type
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
            "train": best_model_evaluation["train"]["metrics"],
            "val": best_model_evaluation["val"]["metrics"],
            "test": best_model_evaluation["test"]["metrics"]
        }

    def __avg_iteration_metrics(self, cv_metrics: list, save: bool = False, inplace: bool = False) -> Union[dict, None]:
        """
        Computes the average metrics for the current CV iteration
        :param cv_metrics: the list of metrics for each processed fold of the CV iteration
        :param save: whether or not to save to file the average metrics
        :param inplace: whether or not to return the average metrics
        """
        avg_scores = {}
        for set_type in ["train", "val", "test"]:
            avg_scores[set_type] = self.__merge_metrics(cv_metrics, set_type)
            print("\n Average {} metrics: \n".format(set_type))
            for metric, value in avg_scores[set_type].items():
                print(("\t - {} " + "".join(["."] * (15 - len(metric))) + " : {}").format(metric, value))

        if save:
            Params.save(avg_scores, os.path.join(self.__paths_to_results["metrics"], "cv_average.json"))

        if not inplace:
            return avg_scores

    def __create_paths_to_results(self):
        """
        Creates the paths to the "metrics", "models", "preds" and "plots" folder for the current experiment.
        If multiple iterations of CV are performed, the nesting of the directories is the following:
        + cv_type_network_type_timestamp
            + seed_n
                + iteration_n
                    - metrics
                    - models
                    - plots
                    - preds
        Otherwise, the iteration_n folder is skipped and the sub-folders belong to the seed_n directory
        """
        path_to_main_dir = os.path.join(self.__path_to_results, "seed_" + str(self.__seed))

        if self.__iteration is not None:
            path_to_main_dir = os.path.join(path_to_main_dir, "iteration_" + str(self.__iteration))

        self.__paths_to_results = {
            "metrics": os.path.join(path_to_main_dir, "metrics"),
            "models": os.path.join(path_to_main_dir, "models"),
            "preds": os.path.join(path_to_main_dir, "preds"),
            "plots": os.path.join(path_to_main_dir, "plots")
        }

        for path in self.__paths_to_results.values():
            os.makedirs(path)

    def validate(self, seed: int, iteration: int = None):
        """
        Performs an iteration of CV for the given random seed
        :param seed: the seed number of the CV
        :param iteration: the iteration number of the CV (None for single-iterations experiments)
        """
        self.__seed, self.__iteration = seed, iteration
        self.__create_paths_to_results()

        cv_metrics, folds_times = [], []
        plotter = Plotter(self.__paths_to_results["plots"])
        zero_time = time.time()

        k = self.__data_manager.get_k()

        for fold in range(1, k + 1):
            fold_heading = "\n * Processing fold {} / {} - seed {} ".format(fold, k, self.__seed)
            fold_heading += "* \n" if self.__iteration is None else "- iteration {} * \n".format(self.__iteration)
            print(fold_heading)

            model_name = self.__train_params["network_type"] + "_fold_" + str(fold)
            path_to_best_model = os.path.join(self.__paths_to_results["models"], model_name + ".pth")
            trainer = Trainer(self.__train_params, path_to_best_model)

            data = self.__data_manager.load_split(fold)

            start_time = time.time()
            model, metrics = trainer.train(self.__data_manager.load_split(fold))
            end_time = time.time()

            best_model_evaluation = self.__evaluator.evaluate_model(data, model, path_to_best_model)

            print("\n *** Finished iteration {} of CV! ***".format(fold))

            print("\n Saving metrics...")
            best_metrics = self.__fetch_best_model_metrics(best_model_evaluation)
            Params.save(best_metrics, os.path.join(self.__paths_to_results["metrics"], "fold_" + str(fold) + ".json"))
            cv_metrics.append(best_metrics)
            print("-> Metrics saved!")

            print("\n Saving preds...")
            Params.save_experiment_preds(best_model_evaluation, self.__paths_to_results["preds"], fold)
            print("-> Predictions saved!")

            print("\n Saving plots...")
            plotter.plot_metrics(metrics, fold)
            print("-> Plots saved!")

            self.__avg_iteration_metrics(cv_metrics, inplace=True)

            folds_times.append((start_time - end_time) / 60)
            estimated_time = (np.mean(np.array(folds_times)) * (k - fold))
            print("\n Time overview: \n")
            print("\t - Time to train fold ............. : {:.2f}m".format((end_time - start_time) / 60))
            print("\t - Elapsed time CV time: .......... : {:.2f}m".format((end_time - zero_time) / 60))
            print("\t - Estimated time of completion ... : {:.2f}m".format(estimated_time))
            print("\n----------------------------------------------------------------\n")

        return self.__avg_iteration_metrics(cv_metrics, save=True)["test"]
