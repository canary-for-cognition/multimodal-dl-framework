import os
import re
import string
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from preprocessing.classes.base.Preprocessor import Preprocessor


class TextPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__("text")
        self.__labels = self._paths.get_labels()

        self.__paths_to_text = self._paths.create_paths(self._params["path_to_src"])

        self.__path_to_csv = os.path.join("assets", "cookie_theft_text.csv")

        # To download the tokenizer run the command: "python -m spacy download en_core_web_sm"
        self.__tokenizer = spacy.load('en_core_web_sm').tokenizer
        self.__stemmer = SnowballStemmer(language='english').stem

    def __tokenize(self, text: str):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text_without_punctuation = regex.sub(" ", text.lower())
        return [token.text for token in self.__tokenizer(text_without_punctuation) if token.text.strip()]

    def __stem(self, tokens: List) -> List:
        return [self.__stemmer(token) for token in tokens]

    @staticmethod
    def __update_max_statistics(sentences: List, max_words: int, max_sentences: int) -> Tuple:
        max_file_words, max_file_sentences = len(max(sentences, key=len)), len(sentences)
        max_words = max_file_words if max_file_words > max_words else max_words
        max_sentences = max_file_sentences if max_file_sentences > max_sentences else max_sentences
        return max_words, max_sentences

    def __extract_vocabulary(self) -> Tuple:
        positive_vocabulary = self.__analyze_vocabulary(self.__paths_to_text["pos"])
        negative_vocabulary = self.__analyze_vocabulary(self.__paths_to_text["neg"])
        vocabulary = positive_vocabulary["vocabulary"].union(negative_vocabulary["vocabulary"])
        max_words = max(positive_vocabulary["max_words"], negative_vocabulary["max_words"])
        max_sentences = max(positive_vocabulary["max_sentences"], negative_vocabulary["max_sentences"])

        words_distribution = list(set(positive_vocabulary["num_words"] + negative_vocabulary["num_words"]))
        print(words_distribution)
        plt.hist(words_distribution, bins=len(words_distribution))
        plt.title("Words distribution")
        plt.show()

        num_words = sum(positive_vocabulary["num_words"]) + sum(negative_vocabulary["num_words"])
        num_items = positive_vocabulary["num_items"] + negative_vocabulary["num_items"]
        avg_words = num_words // num_items

        return vocabulary, max_words, max_sentences, avg_words

    def __get_file_tokens(self, path_to_data: str) -> List:
        file_tokens = []
        for line in open(os.path.join(path_to_data)):
            tokens = self.__tokenize(line)
            if tokens:
                file_tokens += [self.__stem(tokens)]
        return file_tokens

    def __analyze_vocabulary(self, path_to_data: str) -> Dict:
        vocabulary, max_words, max_sentences, num_words = [], 0, 0, []
        text_files = os.listdir(path_to_data)
        for filename in tqdm(text_files, desc="Extracting vocabulary from files at {}".format(path_to_data)):
            file_tokens = self.__get_file_tokens(os.path.join(path_to_data, filename))
            max_words, max_sentences = self.__update_max_statistics(file_tokens, max_words, max_sentences)
            num_words += [sum([len(sentence) for sentence in file_tokens])]
            vocabulary += file_tokens

        return {
            "vocabulary": set([token for tokens in vocabulary for token in tokens]),
            "max_sentences": max_sentences,
            "max_words": max_words,
            "num_words": num_words,
            "num_items": len(text_files)
        }

    @staticmethod
    def __merge_data(pos_data: Dict, neg_data: Dict):
        for k in pos_data.keys():
            pos_data[k] += neg_data[k]
        return pos_data

    @staticmethod
    def __pad_tokens(sentences: List, max_words: int, max_sentences: int) -> List:
        for i, sentence in enumerate(sentences):
            if len(sentence) < max_words:
                sentences[i] += [0] * (max_words - len(sentence))
        if len(sentences) < max_sentences:
            sentences += [[0] * max_words] * (max_sentences - len(sentences))
        return sentences

    def __get_sentences_tokens(self, word2index: Dict, path_to_data: str) -> List:
        sentences_tokens = []
        for line in open(os.path.join(path_to_data)):
            tokens = self.__tokenize(line)
            if tokens:
                sentences_tokens += [[word2index[token] for token in self.__stem(tokens)]]
        return sentences_tokens

    def __process_data(self,
                       word2index: Dict,
                       path_to_text: str,
                       label: str,
                       max_words: int,
                       max_sentences: int) -> Dict:

        parsed_data = {"ids": [], "sentences": [], "tokens": [], "labels": []}

        for filename in tqdm(os.listdir(path_to_text), desc="Processing data at {}".format(path_to_text)):
            sentences_tokens = self.__get_sentences_tokens(word2index, os.path.join(path_to_text, filename))
            padded_tokens = self.__pad_tokens(sentences_tokens, max_words, max_sentences)
            parsed_data["ids"] += [filename.rstrip(".txt")]
            parsed_data["tokens"] += [padded_tokens]
            parsed_data["labels"] += [label]

        return parsed_data

    def __get_processed_data(self, vocabulary: List, max_words: int, max_sentences: int) -> Dict:
        word2index = {word: i + 1 for i, word in enumerate(vocabulary)}

        pos_data = self.__process_data(word2index,
                                       self.__paths_to_text["pos"],
                                       self.__labels["positive"],
                                       max_words,
                                       max_sentences)

        neg_data = self.__process_data(word2index,
                                       self.__paths_to_text["neg"],
                                       self.__labels["negative"],
                                       max_words,
                                       max_sentences)

        return self.__merge_data(pos_data, neg_data)

    def run(self):
        print("\n Extracting vocabulary... \n")
        vocabulary, max_words, max_sentences, avg_words = self.__extract_vocabulary()
        print("\n Analysis of data: \n\n"
              "\t - Number of different words in vocabulary ... : {} \n"
              "\t - Max number of words in a sentence ......... : {} \n"
              "\t - Max number of sentences in a sample ....... : {} \n"
              "\t - Avg number of words ....................... : {} \n"
              .format(len(vocabulary), max_words, max_sentences, avg_words))
        exit()
        print("\n Processing data items... \n")
        data = self.__get_processed_data(vocabulary, max_words, max_sentences)
        data = pd.DataFrame({
            "pid": data["ids"],
            "label": data["labels"],
            "tokens": data["tokens"]
        })

        print("\n Writing text.csv at {}... \n".format(self.__path_to_csv))
        print(data.head())
        data.to_csv(self.__path_to_csv, index=False)
