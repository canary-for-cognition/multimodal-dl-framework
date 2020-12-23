from typing import Callable

from classifier.classes.data.groupers.AlzheimerGrouper import AlzheimerGrouper
from classifier.classes.data.groupers.CognitiveAbilitiesGrouper import CognitiveAbilitiesGrouper
from classifier.classes.data.groupers.ConfusionGrouper import ConfusionGrouper


class GrouperFactory:
    groupers_maps = {
        "confusion": ConfusionGrouper,
        "alzheimer": AlzheimerGrouper,
        "cognitive_abilities": CognitiveAbilitiesGrouper
    }

    def __check_implementation(self, grouper_type: str):
        if grouper_type not in self.groupers_maps.keys():
            raise ValueError("Grouper for {} is not implemented!".format(grouper_type))

    def get_grouper(self, grouper_type: str) -> Callable:
        self.__check_implementation(grouper_type)
        return self.groupers_maps[grouper_type].group

    def get_group_info_extractor(self, grouper_type: str) -> callable:
        self.__check_implementation(grouper_type)
        return self.groupers_maps[grouper_type].get_group_info
