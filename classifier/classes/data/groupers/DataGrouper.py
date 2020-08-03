import pandas as pd

from classifier.classes.factories.GrouperFactory import GrouperFactory


class DataGrouper:

    def __init__(self, dataset_type: str):
        """
        :param dataset_type: the dataset which the data groups are computed for
        """
        self.__grouper = GrouperFactory().get_grouper(dataset_type)

    def get_groups(self, data: pd.DataFrame, group_type: str = "item_id") -> tuple:
        """
        Computes the data groups for the given dataset type
        :param data: the data to be grouped
        :param group_type: the criterion with respect to which data must be grouped
        :return: the grouped data and labels
        """
        negative_group = self.__grouper(data, label=0, group_type=group_type)
        positive_group = self.__grouper(data, label=1, group_type=group_type)
        groups = negative_group + positive_group
        groups_labels = [0 for _ in range(len(negative_group))] + [1 for _ in range(len(positive_group))]
        return groups, groups_labels
