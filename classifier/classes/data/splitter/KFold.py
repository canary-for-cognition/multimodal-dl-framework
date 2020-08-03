import itertools
import os
import random
import shutil

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tqdm import tqdm

from classifier.classes.data.Dataset import Dataset
from classifier.classes.data.groupers.DataGrouper import DataGrouper


class KFold:

    def __init__(self, cv_params: dict, dataset: Dataset):
        """
        :param cv_params: the params related to the CV
        :param dataset: an instance of the selected dataset type
        """
        self.__dataset = dataset
        self.__data = dataset.get_data()
        self.__dataset_type = dataset.get_dataset_type()
        self.__negative_class, self.__positive_class = dataset.get_classes()
        self.__data_grouper = DataGrouper(self.__dataset_type)

        self.__path_to_split = os.path.join(self.__dataset.get_path_to_dataset_folder(), "split")
        self.__path_to_folds = os.path.join(self.__path_to_split, "folds")
        os.makedirs(self.__path_to_folds, exist_ok=True)

        if cv_params["type"] == "leave_one_out":
            groups, groups_labels = self.__data_grouper.get_groups(self.__data)
            self.__k = len(set(groups))
        else:
            self.__k = cv_params["k"]

        self.__downscale = cv_params["downscale"]
        self.__reload_train_test = cv_params["reload_split"]["training_test"]
        self.__force_deletion = cv_params["force_deletion"]
        self.__test_size, self.__validation_size = cv_params["test_size"], cv_params["validation_size"]

    def get_k(self) -> int:
        return self.__k

    def get_path_to_folds(self) -> str:
        return self.__path_to_folds

    def delete_old_data(self, path_to_data):
        """
        Deletes the existing data stored at the given path. If the force_deletion flag is set to true,
        the user will not be prompted to confirm the deletion
        :param path_to_data:
        """
        if not os.path.exists(path_to_data):
            print("\n No existing data to delete! \n")
        else:
            if not self.__force_deletion:
                user_check = " Existing data at will be deleted at {}. \n\n Continue? (y/n) \n".format(path_to_data)
                if input(user_check) != "y":
                    exit()
            print("\n Deleting old data..." if not self.__force_deletion else "\n Force deleting old data...")
            shutil.rmtree(path_to_data)
            print("\n Old data deleted successfully! \n")

    def split_on_training(self):
        """
        Splits the training set in K folds (i.e. training-validation split, using always the same test set)
        """
        print("\n Splitting the training set into {} folds... \n".format(self.__k))
        self.__training_test_split(self.__test_size, self.__reload_train_test)

        training_set = self.__dataset.create_dataset(os.path.join(self.__path_to_split, "training"))
        test_set = self.__dataset.create_dataset(os.path.join(self.__path_to_split, "test"))
        test_set = list(zip(test_set["path"], test_set["label"]))

        groups, groups_labels = self.__data_grouper.get_groups(training_set)
        k_fold = GroupKFold(n_splits=self.__k).split(X=training_set, y=groups_labels, groups=groups)

        for i, (train, val) in enumerate(k_fold):
            print("\n Writing fold {}/{}... \n".format(i + 1, self.__k))
            self.__create_fold(num_fold=i,
                               training=[(training_set.iloc[i]["path"], groups_labels[i]) for i in train],
                               validation=[(training_set.iloc[i]["path"], groups_labels[i]) for i in val],
                               test=test_set)

    def split_on_dataset(self):
        """
        Splits the dataset in K folds (i.e. training-test splits, using always different test sets)
        """
        print("\n Splitting the data into {} folds... \n".format(self.__k))

        groups, groups_labels = self.__data_grouper.get_groups(self.__data)
        k_fold = GroupKFold(n_splits=self.__k).split(X=self.__data, y=groups_labels, groups=groups)

        training_folds, test_folds = [], []
        for i, (training, test) in tqdm(enumerate(k_fold), desc="Creation of training-test split"):
            test_folds += [[(self.__data.iloc[i]["path"], groups_labels[i]) for i in test]]
            training_folds += [[(self.__data.iloc[i]["path"], groups_labels[i]) for i in training]]

        training_folds, validation_folds = self.__split_training_folds(training_folds)
        for i, (training_set, validation_set, test_set) in enumerate(zip(training_folds, validation_folds, test_folds)):
            print("\n Writing fold {}/{}... \n".format(i + 1, self.__k))
            self.__create_fold(num_fold=i, training=training_set, validation=validation_set, test=test_set)

    def split_from_metadata(self, split_info: pd.DataFrame):
        """
        Generates the K splits based on the information stored in a CSV file
        :param split_info: the split metadata info (i.e. the items belonging to each set for the split)
        """
        training_neg = self.__fetch_items_from_metadata(split_info["train_neg"], 0)
        training_pos = self.__fetch_items_from_metadata(split_info["train_pos"], 1)
        training_folds = [neg + pos for (neg, pos) in zip(training_neg, training_pos)]
        training_data, validation_data = self.__split_training_folds(training_folds)

        test_neg = self.__fetch_items_from_metadata(split_info["test_neg"], 0)
        test_pos = self.__fetch_items_from_metadata(split_info["test_pos"], 1)
        test_data = [neg + pos for (neg, pos) in zip(test_neg, test_pos)]

        for i, (training_set, validation_set, test_set) in enumerate(zip(training_data, validation_data, test_data)):
            self.__create_fold(num_fold=i, training=training_set, validation=validation_set, test=test_set)

    def __build_class_directories(self, path_to_source: str, paths_to_data: dict, set_type: str):
        """
        Builds the local positive and negative directories for a given data source
        :param path_to_source: the path to the source data
        :param paths_to_data: the paths to the data items
        :param set_type: the type of data (i.e. training, validation, test)
        """
        path_to_negative = os.path.join(path_to_source, self.__negative_class)
        path_to_positive = os.path.join(path_to_source, self.__positive_class)

        os.makedirs(path_to_negative)
        os.makedirs(path_to_positive)

        print("\n Copy of {} items... \n".format(set_type))

        description = "Copy of {} items".format(self.__negative_class)
        for data_item in tqdm(paths_to_data[self.__negative_class], desc=description):
            shutil.copy(src=data_item, dst=path_to_negative)

        description = "Copy of {} items".format(self.__positive_class)
        for data_item in tqdm(paths_to_data[self.__positive_class], desc=description):
            shutil.copy(src=data_item, dst=path_to_positive)

    def __build_local_split(self,
                            split_data: dict,
                            path_to_training: str = "",
                            path_to_validation: str = "",
                            path_to_test: str = ""):
        """
        Builds the local directories with the split data
        :param split_data: the paths to the data items split into training, validation and test
        :param path_to_training: the path to the training folder of the split
        :param path_to_validation: the path to the validation folder of the split
        :param path_to_test: the path to the test folder of the split
        """
        paths = [path_to_training, path_to_validation, path_to_test]
        set_types = ["training", "validation", "test"]
        for path_to_set, set_type in zip(paths, set_types):
            if path_to_set:
                self.__build_class_directories(path_to_set, split_data[set_type], set_type)

    def __fetch_split_data(self, training: list = None, validation: list = None, test: list = None) -> dict:
        """
        Splits the input data into negative and positive items and store them in a dict
        :param training: the training data as a (item, label) list
        :param test: the test data as a (item, label) list
        :return: a dict containing split data as "training" / "test" and "pos" / "neg"
        """
        split = {}
        for data, set_type in zip([training, validation, test], ["training", "validation", "test"]):
            if data is not None:
                split[set_type] = {
                    self.__negative_class: [item for (item, label) in data if label == 0],
                    self.__positive_class: [item for (item, label) in data if label == 1]
                }
        return split

    def __split_training_folds(self, training_folds: list):
        """
        Splits the input training folds into training and validation
        :param training_folds: a list of lists of training items
        :return: the training folds split into training and validation
        """
        training_data, validation_data = [], []

        for fold in tqdm(training_folds, desc="Creation of training-validation split"):
            data = {"path": [], "item_id": [], "label": []}
            for (path, label) in fold:
                data["path"] += [path]
                data["item_id"] += [path.split(os.sep)[-1].split(".")[0]]
                data["label"] += [label]

            split_data = self.__fetch_train_test_split(pd.DataFrame(data), self.__validation_size)

            training_pos = [(item, 1) for item in split_data["training"][self.__positive_class]]
            training_neg = [(item, 0) for item in split_data["training"][self.__negative_class]]
            if self.__downscale:
                if len(training_pos) < len(training_neg):
                    training_neg = random.sample(training_neg, k=len(training_pos))
                else:
                    training_pos = random.sample(training_pos, k=len(training_neg))
            training_data += [training_pos + training_neg]

            validation_pos = [(item, 1) for item in split_data["test"][self.__positive_class]]
            validation_neg = [(item, 0) for item in split_data["test"][self.__negative_class]]
            validation_data += [validation_pos + validation_neg]

        return training_data, validation_data

    def __create_fold(self, num_fold: int, training: list, validation: list, test: list):
        """
        Builds the local split for a single fold
        :param num_fold: the number corresponding to the current fold
        :param training: the training data for the current fold
        :param validation: the validation data for the current fold
        :param test: the test data for the current fold
        """
        print("\n Generating fold {}... \n".format(num_fold + 1))
        path_to_fold = os.path.join(self.__path_to_folds, "fold_" + str(num_fold + 1))
        self.__build_local_split(split_data=self.__fetch_split_data(training, validation, test),
                                 path_to_training=os.path.join(path_to_fold, "training"),
                                 path_to_validation=os.path.join(path_to_fold, "validation"),
                                 path_to_test=os.path.join(path_to_fold, "test"))

    def __training_test_split(self, test_percentage: float = 0.3, reload_data: bool = False):
        """
        Splits the data into training and test loaders
        :param test_percentage: the percentage of elements of the data to be used for testing
        :param reload_data: whether or not to reload existing data
        """
        path_to_training = os.path.join(self.__path_to_split, "training")
        path_to_test = os.path.join(self.__path_to_split, "test")

        if reload_data:
            print("\n Loading existing training and test split at {}... \n".format(self.__path_to_split))
        else:
            print("\n Splitting the data into training and test... \n")
            self.delete_old_data(path_to_training)
            self.delete_old_data(path_to_test)
            self.delete_old_data(self.__path_to_folds)
            self.__build_local_split(split_data=self.__fetch_train_test_split(self.__data, test_percentage),
                                     path_to_training=path_to_training,
                                     path_to_test=path_to_test)

        print("\n Dataset split overview: \n")
        training_data = self.__dataset.create_dataset_folder(path_to_training)
        self.__dataset.print_data_loader(data_loader=self.__dataset.create_data_loader(training_data),
                                         percentage=1 - test_percentage,
                                         data_type="training")

        test_data = self.__dataset.create_dataset_folder(path_to_test)
        self.__dataset.print_data_loader(data_loader=self.__dataset.create_data_loader(test_data),
                                         percentage=test_percentage,
                                         data_type="test")

    def __fetch_train_test_split(self, data: pd.DataFrame, test_size: float) -> dict:
        """
        Computes the training and test sets of items taking into account groups
        :param data: the data to be split
        :param test_size: the percentage size of the test set
        :return: the data split into train and test
        """
        groups, groups_labels = self.__data_grouper.get_groups(data)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
        train, test = [i for i in gss.split(X=data, y=groups_labels, groups=groups)][0]
        return self.__fetch_split_data(training=[(data.iloc[i]["path"], groups_labels[i]) for i in train],
                                       test=[(data.iloc[i]["path"], groups_labels[i]) for i in test])

    def __fetch_items_from_metadata(self, data: list, label_idx: int) -> list:
        """
        Creates a fold of items with the given label from a series of lists of items stored into the
        column of a csv file.  Note that each user owns 4 data items
        :param data: the column of the csv file
        :param label_idx: the index of the label of the data in the given column
        :return: the restored folds
        """
        restored_folds = []
        label_txt = self.__positive_class if label_idx else self.__negative_class
        file_format = self.__dataset.get_file_format()
        base_path = os.path.join(self.__dataset.get_path_to_main_modality(), label_txt)
        augmented = self.__dataset.is_augmented()

        for i, items in enumerate(data):
            fold = []
            for item in items.strip().split(" "):
                if augmented:
                    for n in itertools.count(start=1):
                        path_to_item = os.path.join(base_path, item + "-" + str(n) + file_format)
                        if os.path.isfile(path_to_item):
                            fold.append((path_to_item, label_idx))
                        else:
                            break
                else:
                    path_to_item = os.path.join(base_path, item + file_format)
                    fold.append((path_to_item, label_idx))

            restored_folds.append(fold)

        return restored_folds
