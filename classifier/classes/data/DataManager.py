import os
import pprint
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from termcolor import colored
from torch.utils.data import DataLoader

from classifier.classes.data.Dataset import Dataset
from classifier.classes.factories.GrouperFactory import GroupingPolicyFactory
from classifier.classes.factories.LoaderFactory import LoaderFactory


class DataManager:

    def __init__(self, data_params: dict, network_type: str):
        self.__path_to_main_modality = data_params["dataset"]["paths"]["main_modality"]
        self.__grouper = GroupingPolicyFactory().get(data_params["dataset"]["name"])
        self.__down_sample_rate = data_params["cv"]["down_sample_rate"]
        self.__k = data_params["cv"]["k"]
        self.__batch_size = data_params["batch_size"]
        self.__loader = LoaderFactory().get(network_type)
        self.__split_data = []
        self.__data = self.__read_data()

    def get_k(self) -> int:
        return self.__k

    def get_group(self, filename: str) -> str:
        return self.__grouper.group(filename)

    def __read_data(self) -> dict:
        data = {}
        for class_dir in os.listdir(self.__path_to_main_modality):
            label = class_dir.split('_')[0]
            path_to_class_dir = os.path.join(self.__path_to_main_modality, class_dir)
            items = os.listdir(path_to_class_dir)
            data[label] = {}
            data[label]['x_paths'] = np.array([os.path.join(path_to_class_dir, item) for item in items])
            data[label]['groups'] = np.array([self.get_group(item) for item in items])
            data[label]['y'] = np.array([int(label)] * len(data[label]['x_paths']))
        return data

    def __generate_split(self):

        print("\n Generating new splits...")

        gss = GroupShuffleSplit(n_splits=self.__k, test_size=1 / self.__k)

        for i in range(self.__k):

            x_train_paths, x_val_paths, x_test_paths = [], [], []
            y_train, y_valid, y_test = [], [], []

            for label in self.__data.keys():
                x_paths, y = self.__data[label]['x_paths'], self.__data[label]['y']
                groups = self.__data[label]['groups']

                train_val, test = list(gss.split(x_paths, y, groups))[i]
                train, val = list(gss.split(x_paths[train_val], y[train_val], groups[train_val]))[i]

                x_train_paths.extend(list(x_paths[train_val][train])), y_train.extend(list(y[train_val][train]))
                x_val_paths.extend(list(x_paths[train_val][val])), y_valid.extend(list(y[train_val][val]))
                x_test_paths.extend(list(x_paths[test])), y_test.extend(list(y[test]))

            self.__split_data.append({
                'train': (x_train_paths, y_train),
                'val': (x_val_paths, y_valid),
                'test': (x_test_paths, y_test)
            })

    def split(self, path_to_metadata: str = ""):
        if path_to_metadata:
            self.__reload_split()
        else:
            self.__generate_split()
        self.__print_split_info()

    def __reload_split(self):
        pass

    def __print_split_info(self):
        """ Shows how the data has been split_data in each fold """

        split_info = []
        for fold_paths in self.__split_data:
            split_info.append({
                'train': list(set([self.get_group(item.split(os.sep)[-1]) for item in fold_paths['train'][0]])),
                'val': list(set([self.get_group(item.split(os.sep)[-1]) for item in fold_paths['val'][0]])),
                'test': list(set([self.get_group(item.split(os.sep)[-1]) for item in fold_paths['test'][0]]))
            })

        print("\n..............................................\n")
        print("Split info overview:\n")
        pp = pprint.PrettyPrinter(compact=True)
        for fold in range(self.__k):
            print(colored(f'fold {fold}: ', 'blue'))
            pp.pprint(split_info[fold])
            print('\n')
        print("\n..............................................\n")

    def __down_sample(self, inputs: list, targets: list) -> tuple:
        data = {0: [], 1: []}
        for (x, y) in zip(inputs, targets):
            data[y] += [(x, y)]

        neg_size, pos_size = len(data[0]), len(data[1])
        if neg_size > pos_size:
            data[0] = random.sample(data[0], k=pos_size * self.__down_sample_rate)
        else:
            data[1] = random.sample(data[1], k=neg_size * self.__down_sample_rate)

        inputs, targets = zip(*data[0] + data[1])

        return inputs, targets

    def load_split(self, fold: int) -> dict:
        """ Loads the data based on the fold paths """

        fold_paths = self.__split_data[fold]

        x_train_paths, y_train = fold_paths['train']
        x_train_paths, y_train = self.__down_sample(x_train_paths, y_train)

        x_val_paths, y_valid = fold_paths['val']
        x_test_paths, y_test = fold_paths['test']

        print("\n..............................................\n")
        print("Split size overview:\n")
        for set_type, y in {"train": y_train, "val": y_valid, "test": y_test}.items():
            num_pos = sum(y)
            num_neg = len(y) - num_pos
            print("\t * {}: [ Pos: {} | Neg: {} ]".format(set_type.upper(), num_pos, num_neg))
        print("\n..............................................\n")

        return {
            'train': DataLoader(Dataset(x_train_paths, y_train, self.__loader), self.__batch_size, shuffle=True),
            'val': DataLoader(Dataset(x_val_paths, y_valid, self.__loader), self.__batch_size),
            'test': DataLoader(Dataset(x_test_paths, y_test, self.__loader), self.__batch_size)
        }

    def save_split_to_file(self, path_to_saved_folds: str):
        pd.DataFrame(self.__split_data).to_csv(path_to_saved_folds, index=False)
