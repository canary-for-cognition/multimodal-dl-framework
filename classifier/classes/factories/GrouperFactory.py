from classifier.classes.data.groupers.AlzheimerGroupingPolicy import AlzheimerGroupingPolicy
from classifier.classes.data.groupers.CognitiveAbilitiesGroupingPolicy import CognitiveAbilitiesGroupingPolicy
from classifier.classes.data.groupers.ConfusionGroupingPolicy import ConfusionGroupingPolicy
from classifier.classes.data.groupers.GroupingPolicy import GroupingPolicy


class GroupingPolicyFactory:
    policies_map = {
        "confusion": ConfusionGroupingPolicy,
        "alzheimer": AlzheimerGroupingPolicy,
        "cognitive_abilities": CognitiveAbilitiesGroupingPolicy
    }

    def get(self, grouper_type: str) -> GroupingPolicy:
        if grouper_type not in self.policies_map.keys():
            raise ValueError("Grouper for {} is not implemented!".format(grouper_type))
        return self.policies_map[grouper_type]()
