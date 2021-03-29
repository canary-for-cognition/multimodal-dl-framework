import json
import os
from typing import Dict


class Params:

    @staticmethod
    def load_dataset_params() -> Dict:
        """
        Loads the dataset parameters stored in the config/dataset.json file
        :return: the loaded parameters in a Dict
        """
        return json.load(open(os.path.join("params", "dataset.json"), "r"))

    @staticmethod
    def load_generator_params(generator_type: str) -> Dict:
        """
        Loads the generator parameters stored in the config/generators/generator_type.json file
        :return: the loaded parameters in a Dict
        """
        return json.load(open(os.path.join("params", "generators", generator_type + ".json"), "r"))

    @staticmethod
    def load_preprocessor_params(preprocessor_type: str) -> Dict:
        """
        Loads the preprocessor parameters stored in the config/preprocessors/preprocessor_type.json file
        :return: the loaded parameters in a Dict
        """
        return json.load(open(os.path.join("params", "preprocessors", preprocessor_type + ".json"), "r"))
