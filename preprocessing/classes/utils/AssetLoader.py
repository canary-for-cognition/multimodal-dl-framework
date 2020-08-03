import os

import yaml


class AssetLoader:

    @staticmethod
    def load_features(features_type: str) -> dict:
        return yaml.safe_load(open(os.path.join("assets", "features", features_type + ".yaml"), mode="rb"))
