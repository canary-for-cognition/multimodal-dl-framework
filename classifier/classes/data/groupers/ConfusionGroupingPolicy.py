from classifier.classes.data.groupers.GroupingPolicy import GroupingPolicy


class ConfusionGroupingPolicy(GroupingPolicy):

    @staticmethod
    def group(filename: str) -> str:
        return filename.split('_')[0][:-1]
