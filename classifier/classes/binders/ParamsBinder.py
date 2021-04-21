class ParamsBinder:
    params_map = {
        "cnn_seq": "cnn",
        "cnn_img": "cnn",
        "cnn_rnn_seq": "cnn_rnn",
        "cnn_rnn_img": "cnn_rnn",
    }

    def get(self, network_type: str) -> str:
        if network_type not in self.params_map.keys():
            return network_type + ".json"
        return self.params_map[network_type] + ".json"
