import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer, BertTokenizer, RobertaTokenizer

from classifier.classes.data.loaders.Loader import Loader
from classifier.classes.utils.Params import Params


class TextLoader(Loader):

    def __init__(self, for_submodule: bool = False):
        super().__init__("text", for_submodule)

        if self._network_type == "transformer":
            network_params = Params.load_network_params(self._network_type)
            self._network_type = network_params["model_type"]
            self.__pretrained_model = network_params["pretrained_model"]
        else:
            self.__word_embeddings = pd.read_csv(os.path.join(self._path_to_modalities["text"], "text.csv"))
            self.__embedding_size = Params.load_modality_params("text")["embedding_size"]

    def __get_tokenizer(self) -> PreTrainedTokenizer:
        tokenizers = {"bert": BertTokenizer, "roberta": RobertaTokenizer}
        return tokenizers[self._network_type].from_pretrained(self.__pretrained_model)

    def __load_bert_encoding(self, path_to_input: str) -> tuple:
        """
        Preprocesses the text converting it into a BERT-friendly format
        :param path_to_input: the path to the data item to be loaded (related to the main modality)
        :return: the indexed text and the relative attention masks
        """
        text = open(self._get_path_to_item(path_to_input)).read()
        tokenizer = self.__get_tokenizer()
        encoding = tokenizer.encode_plus(text,
                                         max_length=self.__embedding_size,
                                         add_special_tokens=True,
                                         return_token_type_ids=False,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_tensors='pt')
        return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()

    def __load_custom_encoding(self, path_to_input: str) -> torch.Tensor:
        item_id = str(path_to_input.split(os.sep)[-1])[:-4]
        encoding = np.array(eval(self.__word_embeddings[self.__word_embeddings["pid"] == item_id]["tokens"].values[0]))
        return torch.from_numpy(encoding)

    def load(self, path_to_input: str) -> Union[torch.Tensor, tuple]:
        """
        Encodes the input for either a pre-trained or custom model
        :param path_to_input: the path to the data item to be encoded (related to the main modality)
        :return: the encoded text for the input data item
        """
        if self._network_type == "transformer":
            return self.__load_bert_encoding(path_to_input)

        return self.__load_custom_encoding(path_to_input)
