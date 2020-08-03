import pandas as pd


class ConfusionGrouper:

    @staticmethod
    def group(data: pd.DataFrame, label: int, group_type: str = "item_id") -> list:
        items = data[data["label"] == label][group_type]
        return [ConfusionGrouper.get_group_info(item_id)["group"] for item_id in items]

    @staticmethod
    def get_group_info(filename: str) -> dict:
        split_info = filename.split("-")
        return {
            "id": filename,
            "frame": split_info[-1] if len(split_info) < 4 else 0,
            "group": split_info[0].split("_")[0][:-1]
        }
