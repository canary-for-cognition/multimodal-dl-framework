import os

import pandas as pd
from tqdm import tqdm

from classifier.classes.data.Dataset import Dataset
from classifier.classes.data.splitter.KFold import KFold


class DataSplitManager:

    def __init__(self, cv_params: dict, dataset: Dataset):
        """
        :param cv_params: the params related to the CV
        :param dataset: an instance of the selected dataset type
        """
        self.__dataset = dataset
        self.__k_fold = KFold(cv_params, dataset)
        self.__path_to_folds = self.__k_fold.get_path_to_folds()

    def get_k(self) -> int:
        items = os.listdir(self.__path_to_folds)
        return len([item for item in items if os.path.isdir(os.path.join(self.__path_to_folds, item))])

    def check_split_availability(self) -> bool:
        """
        Checks if a stored split is present. If not, throws and exception
        :return: true if a stored split is present
        """
        folds = os.listdir(self.__path_to_folds)
        if len(folds) < self.__k_fold.get_k():
            raise ValueError("No stored split available!")
        return True

    def load_fold(self, fold_number: int) -> dict:
        """
        Creates the data loaders for training, validation and test for the given fold
        :param fold_number: the number of the fold to load
        :return: the data loaders for the given fold
        """
        path_to_fold = os.path.join(self.__path_to_folds, "fold_" + str(fold_number))

        set_types = ["training", "validation", "test"]
        data, loaders = {}, {}
        for set_type in set_types:
            data[set_type] = self.__dataset.create_dataset_folder(os.path.join(path_to_fold, set_type))
            loaders[set_type] = self.__dataset.create_data_loader(data[set_type], shuffle=set_type == "training")

        print("\n ** Fold {} ** \n".format(path_to_fold.split("_")[-1]))
        dataset_size = sum([len(data_set) for data_set in data.values()])

        for set_type in set_types:
            self.__dataset.print_data_loader(data_loader=loaders[set_type],
                                             percentage=(len(data[set_type]) / dataset_size) * 100,
                                             data_type=set_type)
        print("..............................................\n")

        return loaders

    def split(self, folds_type: str, save_split: bool = False):
        """
        Splits the data in K folds
        :param folds_type: whether to splits into fold the whole dataset or just the training set
        :param save_split: whether or not to save the generated split on file
        """
        print("\n Generating folds on {} \n".format(folds_type))
        self.__k_fold.delete_old_data(self.__path_to_folds)
        self.__select_folds_type(folds_type)()
        if save_split:
            print("\n Saving split metadata on file... \n")
            self.__save_split_metadata()
            print("\n Saved split metadata on file! \n")

    def split_from_file(self, path_to_cv_metadata: str):
        """
        Generates the K splits based on the information stored in a CSV file
        :param path_to_cv_metadata: the path to the file containing info about the K splits
        """
        print("\n Generating folds from file at {} ... \n".format(path_to_cv_metadata))
        self.__k_fold.delete_old_data(self.__path_to_folds)
        self.__k_fold.split_from_metadata(pd.read_csv(path_to_cv_metadata))

    @staticmethod
    def __fetch_class_items(path_to_items: str, augmented: bool) -> list:
        class_items = []
        for item in os.listdir(path_to_items):
            item_id = item.split(".")[0]
            class_items += [item_id[:-2] if augmented else item_id]
        return class_items

    def __fetch_fold_items(self, path_to_fold: str, pos_class: str, neg_class: str, augmented: bool) -> dict:
        """
        Creates a dictionary containing all the items for all the sets of a given fold
        :param path_to_fold: the path to the root folder of a data fold (containing "training", "test" and "validation")
        :param pos_class: the id of the positive class
        :param neg_class: the id of the negative class
        :param augmented: whether or not the fold items are augmented
        :return: a dictionary containing all the items for all the sets of a given fold
        """
        fold_items = {}
        for set_type in ["training", "validation", "test"]:
            path_to_set = os.path.join(path_to_fold, set_type)
            fold_items[set_type] = {
                "pos": self.__fetch_class_items(os.path.join(path_to_set, pos_class), augmented),
                "neg": self.__fetch_class_items(os.path.join(path_to_set, neg_class), augmented)
            }
        return fold_items

    def __fetch_split_items(self) -> dict:
        """
        Creates a dictionary containing all the items for all the folds of a given split
        :return: a dictionary containing all the items for all the folds of a given split
        """
        split_items = {"training": {"pos": [], "neg": []}, "test": {"pos": [], "neg": []}}
        augmented = self.__dataset.is_augmented()
        neg_class, pos_class = self.__dataset.get_classes()
        for fold in tqdm(os.listdir(self.__path_to_folds), desc="Fetching split items at {}"):
            path_to_fold = os.path.join(self.__path_to_folds, fold)
            fold_items = self.__fetch_fold_items(path_to_fold, pos_class, neg_class, augmented)
            split_items["training"]["pos"] += [fold_items["training"]["pos"] + fold_items["validation"]["pos"]]
            split_items["training"]["neg"] += [fold_items["training"]["neg"] + fold_items["validation"]["neg"]]
            split_items["test"]["pos"] += [fold_items["test"]["pos"]]
            split_items["test"]["neg"] += [fold_items["test"]["neg"]]
        return split_items

    def __save_split_metadata(self):
        """
        Saves the current data split on a CSV file
        """
        path_to_saved_split = os.path.join(self.__path_to_folds, "split_metadata.csv")
        if not os.path.isfile(path_to_saved_split):
            split_items = self.__fetch_split_items()
            split_metadata = pd.DataFrame({
                "train_pos": [" ".join(items) for items in split_items["training"]["pos"]],
                "train_neg": [" ".join(items) for items in split_items["training"]["neg"]],
                "test_pos": [" ".join(items) for items in split_items["test"]["pos"]],
                "test_neg": [" ".join(items) for items in split_items["test"]["neg"]]
            })
            split_metadata.to_csv(path_to_saved_split, index=False)
        else:
            print("\n WARNING: not writing split metadata, file already present at {}\n".format(path_to_saved_split))

    def __select_folds_type(self, folds_type: str) -> callable:
        """
        Selects the function to be used to fold the dataset (creating folds on the whole dataset on just on the
        training set)
        :param folds_type: whether to splits into fold the whole dataset or just the training set
        :return: a splitting function
        """
        folds_types_map = {
            "training": self.__k_fold.split_on_training,
            "dataset": self.__k_fold.split_on_dataset
        }
        if folds_type not in folds_types_map.keys():
            raise ValueError("The selected folds split type {} is not supported!".format(folds_type))
        return folds_types_map[folds_type]
