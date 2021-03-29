import os
from typing import Dict

import yaml


class AssetLoader:

    @staticmethod
    def load_features(features_type: str) -> Dict:
        return yaml.safe_load(open(os.path.join("assets", "features", features_type + ".yaml"), mode="rb"))
