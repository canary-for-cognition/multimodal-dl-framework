import itertools
import os
import random
import shutil

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tqdm import tqdm

from classifier.classes.data.Dataset import Dataset
from classifier.classes.data.groupers.DataGrouper import DataGrouper
from classifier.classes.factories.GrouperFactory import GrouperFactory


class DataManager:

    def __init__(self, cv_params: dict, dataset: Dataset):
        """
        :param cv_params: the params related to the CV
        :param dataset: an instance of the selected dataset type
        """
        self.__dataset = dataset
        self.__data = dataset.get_data()
        self.__neg_class, self.__pos_class = dataset.get_classes()
        self.__data_grouper = DataGrouper(dataset.get_dataset_type())

        self.__path_to_folds = os.path.join(self.__dataset.get_path_to_dataset_folder(), "split", "folds")
        os.makedirs(self.__path_to_folds, exist_ok=True)

        self.__k = cv_params["k"]
        self.__down_sample = cv_params["down_sample"]
        self.__force_deletion = cv_params["force_deletion"]
        self.__val_size = cv_params["val_size"]

    def get_k(self) -> int:
        return len([filename for filename in os.listdir(self.__path_to_folds) if filename.startswith('fold_')])

    def split_available(self) -> bool:
        """
        Checks if a stored split is present. If not, throws and exception
        :return: true if a stored split is present
        """
        if len(os.listdir(self.__path_to_folds)) < self.__k:
            raise ValueError("No stored split available!")
        return True

    def clear_stored_split(self):
        """
        Deletes the existing data stored at the given path. If the force_deletion flag is set to true,
        the user will not be prompted to confirm the deletion
        """
        if not os.path.exists(self.__path_to_folds):
            print("WARNING: no existing data to delete!")
            return

        if not self.__force_deletion:
            user_check = "Existing data at will be deleted at {}. \n\n Continue? (y/n) \n".format(self.__path_to_folds)
            if input(user_check) != "y":
                exit()

        print("\nDeleting old data..." if not self.__force_deletion else "\nForce deleting old data...")
        shutil.rmtree(self.__path_to_folds)
        print("-> Old data deleted successfully! \n")

    def split(self):
        """
        Splits the dataset in K folds (i.e. train-val splits using always different test sets)
        """
        self.clear_stored_split()

        print("\n Splitting the data into {} folds... \n".format(self.__k))
        groups, groups_labels = self.__data_grouper.get_groups(self.__data)
        k_fold = GroupKFold(n_splits=self.__k).split(X=self.__data, y=groups_labels, groups=groups)

        train_folds, test_folds = [], []
        for (train, test) in tqdm(k_fold, desc="Train-test split"):
            test_folds += [[(self.__data.iloc[i]["path"], groups_labels[i]) for i in test]]
            train_folds += [[(self.__data.iloc[i]["path"], groups_labels[i]) for i in train]]

        train_folds, val_folds = self.__train_val_split(train_folds)
        for i, (train_set, val_set, test_set) in enumerate(zip(train_folds, val_folds, test_folds)):
            self.__build_local_split(num_fold=i, train=train_set, val=val_set, test=test_set)

        print("Saving split metadata on file...")
        self.__save_split_metadata()
        print("-> Saved split metadata on file! \n")

    def split_from_metadata(self, path_to_cv_metadata: str):
        """
        Generates the K splits based on the information stored in a CSV file
        :param path_to_cv_metadata: the path to the split metadata info
        """
        print("Generating folds from file at {} ...".format(path_to_cv_metadata))
        self.clear_stored_split()

        split_info = pd.read_csv(path_to_cv_metadata)

        train_neg = self.__fetch_items_from_metadata(split_info["train_neg"], 0)
        train_pos = self.__fetch_items_from_metadata(split_info["train_pos"], 1)
        train_folds = [neg + pos for (neg, pos) in zip(train_neg, train_pos)]
        train_data, val_data = self.__train_val_split(train_folds)

        test_neg = self.__fetch_items_from_metadata(split_info["test_neg"], 0)
        test_pos = self.__fetch_items_from_metadata(split_info["test_pos"], 1)
        test_data = [neg + pos for (neg, pos) in zip(test_neg, test_pos)]

        for i, (train_fold, val_fold, test_fold) in enumerate(zip(train_data, val_data, test_data)):
            self.__build_local_split(num_fold=i, train=train_fold, val=val_fold, test=test_fold)

    def __split_by_class(self, train: list = None, val: list = None, test: list = None) -> dict:
        """
        Splits the input data into neg and pos items and store them in a dict
        :param train: the train data as a (item, label) list
        :param test: the test data as a (item, label) list
        :return: a dict containing split data as "train" / "test" and "pos" / "neg"
        """
        split = {}
        for data, set_type in zip([train, val, test], ["train", "val", "test"]):
            if data:
                split[set_type] = {self.__neg_class: [], self.__pos_class: []}
                for (item, label) in data:
                    split[set_type][self.__pos_class if label else self.__neg_class] += [item]
        return split

    def __train_val_split(self, train_folds: list):
        """
        Splits the input train folds into train and val
        :param train_folds: a list of lists of train items
        :return: the train folds split into train and val
        """
        train_data, val_data = [], []

        for fold in tqdm(train_folds, desc="Training-val split"):

            data = {"path": [], "item_id": [], "label": []}
            for (path, label) in fold:
                data["path"] += [path]
                data["item_id"] += [path.split(os.sep)[-1].split(".")[0]]
                data["label"] += [label]
            data = pd.DataFrame(data)

            groups, groups_labels = self.__data_grouper.get_groups(data)
            gss = GroupShuffleSplit(n_splits=1, test_size=self.__val_size)
            train, val = [i for i in gss.split(X=data, y=groups_labels, groups=groups)][0]
            split_data = self.__split_by_class(train=[(data.iloc[i]["path"], groups_labels[i]) for i in train],
                                               val=[(data.iloc[i]["path"], groups_labels[i]) for i in val])

            train_pos = [(item, 1) for item in split_data["train"][self.__pos_class]]
            train_neg = [(item, 0) for item in split_data["train"][self.__neg_class]]
            if self.__down_sample:
                if len(train_pos) < len(train_neg):
                    train_neg = random.sample(train_neg, k=len(train_pos) * self.__down_sample)
                else:
                    train_pos = random.sample(train_pos, k=len(train_neg) * self.__down_sample)
            train_data += [train_pos + train_neg]

            val_pos = [(item, 1) for item in split_data["val"][self.__pos_class]]
            val_neg = [(item, 0) for item in split_data["val"][self.__neg_class]]
            val_data += [val_pos + val_neg]

        return train_data, val_data

    def __build_local_split(self, num_fold: int, train: list, val: list, test: list):
        """
        Builds the local split for a single fold
        :param num_fold: the number corresponding to the current fold
        :param train: the train data for the current fold
        :param val: the val data for the current fold
        :param test: the test data for the current fold
        """
        print("\n Generating fold {}/{}... \n".format(num_fold + 1, self.__k))
        path_to_fold = os.path.join(self.__path_to_folds, "fold_" + str(num_fold + 1))
        split_data = self.__split_by_class(train, val, test)

        for set_type in ["train", "val", "test"]:
            paths_to_data = split_data[set_type]
            path_to_set_dir = os.path.join(path_to_fold, set_type)

            path_to_neg = os.path.join(path_to_set_dir, self.__neg_class)
            os.makedirs(path_to_neg)

            path_to_pos = os.path.join(path_to_set_dir, self.__pos_class)
            os.makedirs(path_to_pos)

            print("\nCopying {} items...\n".format(set_type))

            for data_item in tqdm(paths_to_data[self.__neg_class], desc=self.__neg_class):
                shutil.copy(src=data_item, dst=path_to_neg)

            for data_item in tqdm(paths_to_data[self.__pos_class], desc=self.__pos_class):
                shutil.copy(src=data_item, dst=path_to_pos)

    def __fetch_items_from_metadata(self, data: list, label_idx: int) -> list:
        """
        Creates a fold of items with the given label from a series of lists of items stored into the
        column of a csv file.  Note that each user owns 4 data items
        :param data: the column of the csv file
        :param label_idx: the index of the label of the data in the given column
        :return: the restored folds
        """
        restored_folds = []
        label_txt = self.__pos_class if label_idx else self.__neg_class
        file_format = self.__dataset.get_file_format()
        base_path = os.path.join(self.__dataset.get_path_to_main_modality(), label_txt)
        augmented = self.__dataset.is_augmented()

        for i, items in enumerate(data):
            fold = []
            for item in items.strip().split(" "):
                if augmented:
                    for n in itertools.count(start=1):
                        path_to_item = os.path.join(base_path, item + "-" + str(n) + file_format)
                        if not os.path.isfile(path_to_item):
                            break
                        fold.append((path_to_item, label_idx))
                else:
                    path_to_item = os.path.join(base_path, item + file_format)
                    fold.append((path_to_item, label_idx))

            restored_folds.append(fold)

        return restored_folds

    def load_split(self, fold_number: int) -> dict:
        """
        Creates the data loaders for train, val and test for the given fold number
        :param fold_number: the number of the fold to load
        :return: the data loaders for the given fold number
        """
        path_to_fold = os.path.join(self.__path_to_folds, "fold_" + str(fold_number))
        print("\n ** Fold {} ** \n".format(fold_number))

        dataset_size = len(self.__dataset)
        data, loaders = {}, {}
        for set_type in ["train", "val", "test"]:
            data[set_type] = self.__dataset.create_dataset_folder(os.path.join(path_to_fold, set_type))
            loaders[set_type] = self.__dataset.create_data_loader(data[set_type], shuffle=set_type == "train")
            set_size = len(data[set_type])
            print("\n{} set ({:.2f}%):\n".format(set_type.upper(), (set_size / dataset_size) * 100))
            print("\t- Total number of items: {}".format(set_size))
            for label in os.listdir(os.path.join(path_to_fold, set_type)):
                print("\t- {}: {}".format(label, len(os.listdir(os.path.join(path_to_fold, set_type, label)))))

        print("\n..............................................\n")

        return loaders

    def __save_split_metadata(self):
        """
        Saves the current data split on a CSV file
        """
        path_to_saved_split = os.path.join(self.__path_to_folds, "metadata.csv")

        if os.path.isfile(path_to_saved_split):
            print("WARNING: not writing split metadata, file already present at {}".format(path_to_saved_split))
            return

        split_data = {"train": {"neg": [], "pos": []}, "test": {"neg": [], "pos": []}}
        group_info_extractor = GrouperFactory().get_group_info_extractor(self.__dataset.get_dataset_type())

        for fold in tqdm(os.listdir(self.__path_to_folds), desc="Fetching split items at {}"):
            path_to_fold = os.path.join(self.__path_to_folds, fold)
            fold_data = {"train": {}, "val": {}, "test": {}}

            for set_type in fold_data.keys():
                path_to_set = os.path.join(path_to_fold, set_type)
                for label in os.listdir(path_to_set):
                    path_to_items = os.path.join(path_to_set, label)
                    label_idx = int(label.split("_")[0])
                    fold_data[set_type][label_idx] = [group_info_extractor(f)["group"] for f in path_to_items]

            split_data["train"]["neg"] += [fold_data["train"][0] + fold_data["val"][0]]
            split_data["train"]["pos"] += [fold_data["train"][1] + fold_data["val"][1]]
            split_data["test"]["neg"] += [fold_data["test"][0]]
            split_data["test"]["pos"] += [fold_data["test"][1]]

        split_metadata = pd.DataFrame({
            "train_pos": [" ".join(items) for items in split_data["train"]["pos"]],
            "train_neg": [" ".join(items) for items in split_data["train"]["neg"]],
            "test_pos": [" ".join(items) for items in split_data["test"]["pos"]],
            "test_neg": [" ".join(items) for items in split_data["test"]["neg"]]
        })
        split_metadata.to_csv(path_to_saved_split, index=False)
