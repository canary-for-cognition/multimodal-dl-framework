import os
import time
from typing import Union, Dict, List

import numpy as np

from classifier.classes.core.Evaluator import Evaluator
from classifier.classes.core.Trainer import Trainer
from classifier.classes.data.DataManager import DataManager
from classifier.classes.utils.Params import Params


class CrossValidator:

    def __init__(self, data_manager: DataManager, path_to_results: str, train_params: Dict):
        """
        :param data_manager: an instance of DataManager to load the folds from the filesystem
        :param path_to_results: the path to the directory with the results for the current experiment
        :param train_params: the params to be submitted to the Trainer instance
        """
        self.__data_manager = data_manager
        self.__path_to_results = path_to_results
        self.__train_params = train_params
        self.__evaluator = Evaluator(train_params["device"])
        self.__paths_to_results = {}

    @staticmethod
    def __merge_metrics(metrics: List, set_type: str) -> Dict:
        """
        Averages the metrics by set type (in ["train", "val", "test"])
        :param metrics: the metrics of each processed fold
        :param set_type: set code in ["train", "val", "test"]
        :return: the input metrics averaged by set type
        """
        return {k: np.array([m[set_type][k] for m in metrics]).mean() for k in metrics[0][set_type].keys()}

    def __avg_metrics(self, cv_metrics: List, save: bool = False, inplace: bool = False) -> Union[Dict, None]:
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

    def __create_paths_to_results(self, seed: int):
        """
        Creates the paths to the "metrics", "models", "preds" and "plots" folder for the current experiment.
        @param seed: the current random seed
        """
        path_to_main_dir = os.path.join(self.__path_to_results, "seed_" + str(seed))

        self.__paths_to_results = {
            "metrics": os.path.join(path_to_main_dir, "metrics"),
            "models": os.path.join(path_to_main_dir, "models"),
            "preds": os.path.join(path_to_main_dir, "preds"),
            "plots": os.path.join(path_to_main_dir, "plots")
        }

        for path in self.__paths_to_results.values():
            os.makedirs(path)

    def validate(self, seed: int) -> Dict:
        """
        Performs an iteration of CV for the given random seed
        :param seed: the seed number of the CV
        """
        self.__create_paths_to_results(seed)

        cv_metrics, folds_times = [], []
        # plotter = Plotter(self.__paths_to_results["plots"])
        zero_time = time.time()

        k = self.__data_manager.get_k()

        for fold in range(k):
            print("\n * Processing fold {} / {} - seed {} * \n".format(fold + 1, k, seed))

            model_name = self.__train_params["network_type"] + "_fold_" + str(fold)
            path_to_best_model = os.path.join(self.__paths_to_results["models"], model_name + ".pth")
            trainer = Trainer(self.__train_params, path_to_best_model)

            data = self.__data_manager.load_split(fold)

            start_time = time.time()
            model, evaluations = trainer.train(data)
            end_time = time.time()

            best_eval = self.__evaluator.evaluate(data, model, path_to_best_model)

            print("\n *** Finished iteration {} of CV! ***".format(fold))

            print("\n Saving metrics...")
            metrics_log = "fold_" + str(fold) + ".json"
            Params.save(best_eval["metrics"], os.path.join(self.__paths_to_results["metrics"], metrics_log))
            cv_metrics.append(best_eval["metrics"])
            print("-> Metrics saved!")

            print("\n Saving preds...")
            Params.save_experiment_preds(best_eval, self.__paths_to_results["preds"], fold + 1)
            print("-> Predictions saved!")

            # print("\n Saving plots...")
            # plotter.plot_metrics(metrics, fold + 1)
            # print("-> Plots saved!")

            self.__avg_metrics(cv_metrics, inplace=True)

            folds_times.append((start_time - end_time) / 60)
            estimated_time = (np.mean(np.array(folds_times)) * (k - fold))
            print("\n Time overview: \n")
            print("\t - Time to train fold ............. : {:.2f}m".format((end_time - start_time) / 60))
            print("\t - Elapsed time CV time: .......... : {:.2f}m".format((end_time - zero_time) / 60))
            print("\t - Estimated time of completion ... : {:.2f}m".format(estimated_time))
            print("\n----------------------------------------------------------------\n")

        return self.__avg_metrics(cv_metrics, save=True)["test"]
