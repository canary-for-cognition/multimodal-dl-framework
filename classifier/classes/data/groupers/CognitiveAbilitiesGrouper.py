import pandas as pd


class CognitiveAbilitiesGrouper:

    @staticmethod
    def group(data: pd.DataFrame, label: int, group_type: str = "item_id") -> list:
        groups = []
        for item_id in data[data["label"] == label][group_type]:
            item_info = CognitiveAbilitiesGrouper.get_group_info(item_id)
            groups.append(item_info["group"])
        return groups

    @staticmethod
    def get_group_info(filename: str) -> dict:
        filename_info = filename.split(".")[0].split("-")
        return {
            "id": filename,
            "group": filename_info[0],
            "frame": filename_info[2]
        }
