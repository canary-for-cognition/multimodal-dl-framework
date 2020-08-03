class ParamsBinder:
    params_map = {
        "sequences_cnn": "cnn",
        "images_cnn": "cnn",
        "sequences_cnn_rnn": "cnn_rnn",
        "images_cnn_rnn": "cnn_rnn",
    }

    def get(self, network_type: str) -> str:
        if network_type not in self.params_map.keys():
            return network_type + ".json"
        return self.params_map[network_type] + ".json"
