from typing import Tuple

from transformers import BertTokenizer, RobertaTokenizer

from classifier.classes.data.loaders.Loader import Loader
from classifier.classes.utils.Params import Params


class TextLoader(Loader):

    def __init__(self, for_submodule: bool = False):
        super().__init__("text", for_submodule)

        network_params = Params.load_network_params(self._network_type)
        self._network_type = network_params["model_type"]
        self.__pretrained_model = network_params["pretrained_model"]
        self.__embedding_size = Params.load_modality_params("text")["embedding_size"]

        tokenizers = {"bert": BertTokenizer, "roberta": RobertaTokenizer}
        self.__tokenizer = tokenizers[self._network_type].from_pretrained(self.__pretrained_model)

    def load(self, path_to_input: str) -> Tuple:
        """
        Preprocesses the text converting it into a BERT-friendly format
        :param path_to_input: the path to the data item to be loaded (related to the main modality)
        :return: the indexed text and the relative attention masks
        """
        encoding = self.__tokenizer.encode_plus(text=open(self._get_path_to_item(path_to_input)).read(),
                                                max_length=self.__embedding_size,
                                                add_special_tokens=True,
                                                return_token_type_ids=False,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')

        input_ids, attention_mask = encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()

        return input_ids, attention_mask
