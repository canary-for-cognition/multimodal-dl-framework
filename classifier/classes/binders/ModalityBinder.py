class ModalityBinder:
    modalities_map = {
        "rnn": "sequences",
        "cnn_seq": "sequences",
        "cnn_rnn_seq": "sequences",
        "cnn_img": "images",
        "cnn_rnn_img": "images",
        "pre_trained_cnn": "images",
        "transformer": "text",
        "vistempnet": ("images", "sequences"),
        "vistextnet": ("images", "text")
    }

    def get(self, network_type: str) -> str:
        if network_type not in self.modalities_map.keys():
            raise ValueError("Network {} is not implemented, could not fetch modality! \n Implemented networks are: {}"
                             .format(network_type, list(self.modalities_map.keys())))
        return self.modalities_map[network_type]
