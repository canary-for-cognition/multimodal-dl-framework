import json
import os


class Params:

    @staticmethod
    def load_dataset_params() -> dict:
        """
        Loads the dataset parameters stored in the config/dataset.json file
        :return: the loaded parameters in a dict
        """
        return json.load(open(os.path.join("params", "dataset.json"), "r"))

    @staticmethod
    def load_generator_params(generator_type: str) -> dict:
        """
        Loads the generator parameters stored in the config/generators/generator_type.json file
        :return: the loaded parameters in a dict
        """
        return json.load(open(os.path.join("params", "generators", generator_type + ".json"), "r"))

    @staticmethod
    def load_preprocessor_params(preprocessor_type: str) -> dict:
        """
        Loads the preprocessor parameters stored in the config/preprocessors/preprocessor_type.json file
        :return: the loaded parameters in a dict
        """
        return json.load(open(os.path.join("params", "preprocessors", preprocessor_type + ".json"), "r"))
