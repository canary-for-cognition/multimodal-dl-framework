import json
import os

import pandas as pd

from classifier.classes.binders.ModalityBinder import ModalityBinder
from classifier.classes.binders.ParamsBinder import ParamsBinder


class Params:

    @staticmethod
    def load() -> tuple:
        experiment_params = Params.load_experiment_params()
        train_params = experiment_params["train"]
        data_params = {
            "dataset": Params.load_dataset_params(experiment_params["dataset_name"]),
            "cv": experiment_params["cv"],
            "batch_size": train_params["batch_size"]
        }
        num_seeds = experiment_params["num_seeds"]
        device_type = experiment_params["device"]
        return train_params, data_params, num_seeds, device_type

    @staticmethod
    def load_experiment_params() -> dict:
        """
        Loads the parameters stored in the params/experiment.json file
        :return: the loaded experiment parameters in a dict
        """
        return json.load(open(os.path.join("params", "experiment.json"), "r"))

    @staticmethod
    def load_modality_params(modality: str) -> dict:
        """
        Loads the parameters stored in the params/modalities/modality.json file
        :param modality: the modality params to be loaded
        :return: the loaded cross val parameters in a dict
        """
        return json.load(open(os.path.join("params", "modalities", modality + ".json"), "r"))

    @staticmethod
    def __load_submodules_params(submodules_map: dict) -> dict:
        """
        Loads the submodules parameters for a multimodal network
        :param submodules_map: the dict containing the modalities and corresponding submodules to be loaded
        :return: the loaded modality-submodule params in a dict
        """
        submodules_params = {}
        for modality, network_type in submodules_map.items():
            module_params = json.load(open(os.path.join("params", "networks", ParamsBinder().get(network_type)), "r"))
            module_params["architecture"] = network_type
            module_params["batch_size"] = Params.load_experiment_params()["train"]["batch_size"]
            module_params["modality"] = Params.load_modality_params(ModalityBinder().get(network_type))
            submodules_params[modality] = module_params
        return submodules_params

    @staticmethod
    def load_network_params(network_type: str) -> dict:
        """
        Loads and preprocesses the parameters stored in the params/modules/network_type.json file
        :param network_type: the type of network to be loaded
        :return: the loaded network parameters in a dict
        """
        network_params = json.load(open(os.path.join("params", "networks", ParamsBinder().get(network_type)), "r"))

        if "submodules" in network_params.keys():
            network_params["submodules"] = Params.__load_submodules_params(network_params["submodules"])
        else:
            network_params["architecture"] = network_type
            network_params["batch_size"] = Params.load_experiment_params()["train"]["batch_size"]
            network_params["modality"] = Params.load_modality_params(ModalityBinder().get(network_type))

        return network_params

    @staticmethod
    def load_dataset_params(dataset_name: str) -> dict:
        """
        Loads the parameters stored in the params/modules/dataset_name.json file merging the paths
        :param dataset_name: the type of data to be loaded
        :return: the loaded data parameters in a dict
        """
        params = json.load(open(os.path.join("params", "dataset", dataset_name + ".json"), "r"))
        params["name"] = dataset_name

        dataset_dir = params["paths"]["dataset_dir"]
        params["paths"]["dataset_metadata"] = os.path.join(dataset_dir, params["paths"]["dataset_metadata"])
        params["paths"]["cv_metadata"] = os.path.join(dataset_dir, params["paths"]["cv_metadata"])
        params["paths"]["main_modality"] = os.path.join(dataset_dir, "modalities", params["paths"]["main_modality"])
        for modality, path_to_modality in params["paths"]["modalities"].items():
            params["paths"][modality] = os.path.join(params["paths"]["dataset_dir"], "modalities", path_to_modality)

        return params

    @staticmethod
    def save(data: dict, path_to_destination):
        """
        Saves the given data into a JSON file at the given destination
        :param data: the data to be saved
        :param path_to_destination: the destination of the file with the saved metrics
        """
        json.dump(data, open(path_to_destination, 'w'), indent=2)

    @staticmethod
    def save_experiment_params(path_to_results: str, network_type: str, dataset_name: str):
        """
        Saves the configuration for the current experiment to file at the given path
        :param path_to_results: the path where to save the configuration of the experiment at
        :param network_type: the type of network used for the experiment
        :param dataset_name: the type of data used for the experiment
        """
        Params.save(Params.load_experiment_params(), os.path.join(path_to_results, "experiment.json"))
        Params.save(Params.load_dataset_params(dataset_name), os.path.join(path_to_results, "data.json"))
        Params.save(Params.load_network_params(network_type), os.path.join(path_to_results, "network_params.json"))

    @staticmethod
    def save_experiment_preds(fold_evaluation: dict, path_to_preds: str, fold_number: int):
        """
        Saves the experiments preds in CSV format at the given path
        :param fold_evaluation: the evaluation data for the given fold, including ground truth and preds
        :param path_to_preds: the path where to store the preds at
        :param fold_number: the number of the fold for generating the name of the file
        """
        for set_type in ["train", "val", "test"]:
            path_to_csv = os.path.join(path_to_preds, "fold_" + str(fold_number) + "_" + set_type + "_preds.csv")
            pd.DataFrame(fold_evaluation["preds"][set_type]).to_csv(path_to_csv, index=False)
