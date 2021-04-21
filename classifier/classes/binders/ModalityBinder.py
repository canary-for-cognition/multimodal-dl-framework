class ModalityBinder:
    modalities_map = {
        "rnn": "sequences",
        "cnn_seq": "sequences",
        "cnn_rnn_seq": "sequences",
        "cnn_img": "images",
        "cnn_rnn_img": "images",
        "pre_trained_cnn": "images",
        "han": "text",
        "transformer": "text",
        "vistempnet": ("images", "sequences"),
        "vistextnet": ("images", "text")
    }

    def get(self, network_type: str) -> str:
        if network_type not in self.modalities_map.keys():
            raise ValueError("Network {} is not implemented, could not fetch modality!".format(network_type))
        return self.modalities_map[network_type]
