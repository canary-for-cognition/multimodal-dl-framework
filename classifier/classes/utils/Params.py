import json
import os

import pandas as pd

from classifier.classes.binders.ModalityBinder import ModalityBinder
from classifier.classes.binders.ParamsBinder import ParamsBinder


class Params:

    @staticmethod
    def load_experiment_params() -> dict:
        """
        Loads the parameters stored in the params/experiment.json file
        :return: the loaded experiment parameters in a dict
        """
        return json.load(open(os.path.join("params", "experiment.json"), "r"))

    @staticmethod
    def load_cv_params() -> dict:
        """
        Loads the parameters stored in the params/cross_validation.json file
        :return: the loaded cross validation parameters in a dict
        """
        return json.load(open(os.path.join("params", "cross_validation.json"), "r"))

    @staticmethod
    def load_modality_params(modality: str) -> dict:
        """
        Loads the parameters stored in the params/modalities/modality.json file
        :param modality: the modality params to be loaded
        :return: the loaded cross validation parameters in a dict
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
            module_params["batch_size"] = Params.load_experiment_params()["training"]["batch_size"]
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
            network_params["batch_size"] = Params.load_experiment_params()["training"]["batch_size"]
            network_params["modality"] = Params.load_modality_params(ModalityBinder().get(network_type))

        return network_params

    @staticmethod
    def load_dataset_params(dataset_type: str) -> dict:
        """
        Loads the parameters stored in the params/modules/dataset_name.json file merging the paths
        :param dataset_type: the type of data to be loaded
        :return: the loaded data parameters in a dict
        """
        params = json.load(open(os.path.join("params", "dataset", dataset_type + ".json"), "r"))

        main_modality = ModalityBinder().get(Params.load_experiment_params()["network_type"])

        params["type"] = dataset_type
        params["main_modality"] = main_modality[0] if isinstance(main_modality, tuple) else main_modality

        dataset_folder = params["paths"]["dataset_folder"]
        params["paths"]["dataset_metadata"] = os.path.join(dataset_folder, params["paths"]["dataset_metadata"])
        params["paths"]["cv_metadata"] = os.path.join(dataset_folder, params["paths"]["cv_metadata"])
        for modality, path_to_modality in params["paths"]["modalities"].items():
            params["paths"][modality] = os.path.join(params["paths"]["dataset_folder"], "modalities", path_to_modality)

        return params

    @staticmethod
    def load_cv_metadata(path_to_cv_metadata: str) -> list:
        """
        Loads the CV metadata, i.e. file(s) describing the splits for the validation procedure
        :param path_to_cv_metadata: the path to the CV metadata (to a single file or folder)
        :return: the CV metadata in a list
        """
        if os.path.isdir(path_to_cv_metadata):
            sorted_cv_files = sorted(os.listdir(path_to_cv_metadata), key=lambda x: int(x.split('.')[0].split('_')[2]))
            return [os.path.join(path_to_cv_metadata, file) for file in sorted_cv_files]
        return [path_to_cv_metadata]

    @staticmethod
    def save(data: dict, path_to_destination):
        """
        Saves the given data into a JSON file at the given destination
        :param data: the data to be saved
        :param path_to_destination: the destination of the file with the saved metrics
        """
        json.dump(data, open(path_to_destination, 'w'), indent=2)

    @staticmethod
    def save_experiment_params(path_to_experiment: str, network_type: str, dataset_type: str):
        """
        Saves the configuration for the current experiment to file at the given path
        :param path_to_experiment: the path where to save the configuration of the experiment at
        :param network_type: the type of network used for the experiment
        :param dataset_type: the type of data used for the experiment
        """
        Params.save(Params.load_experiment_params(), os.path.join(path_to_experiment, "experiment_setting.json"))
        Params.save(Params.load_dataset_params(dataset_type), os.path.join(path_to_experiment, "data.json"))
        Params.save(Params.load_network_params(network_type), os.path.join(path_to_experiment, "network_params.json"))

    @staticmethod
    def save_experiment_predictions(fold_evaluation: dict, path_to_predictions: str, fold_number: int):
        """
        Saves the experiments predictions in CSV format at the given path
        :param fold_evaluation: the evaluation data for the given fold, including ground truth and predictions
        :param path_to_predictions: the path where to store the predictions at
        :param fold_number: the number of the fold for generating the name of the file
        """
        for set_type in ["training", "validation", "test"]:
            predictions = fold_evaluation[set_type]["predictions"]
            predictions_data = {
                "items_ids": predictions["items_ids"],
                "ground_truth": predictions["y_true"],
                "prediction": predictions["y_pred"]
            }
            file_name = "fold_" + str(fold_number) + "_" + set_type + "_predictions.csv"
            pd.DataFrame(predictions_data).to_csv(os.path.join(path_to_predictions, file_name), index=False)
