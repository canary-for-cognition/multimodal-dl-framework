import pandas as pd


class AlzheimerGrouper:

    @staticmethod
    def group(data: pd.DataFrame, label: int, group_type: str = "item_id") -> list:
        groups = []
        for item_id in data[data["label"] == label][group_type]:
            item_info = AlzheimerGrouper.get_group_info(item_id)
            groups.append(item_info["group"])
        return groups

    @staticmethod
    def get_group_info(filename: str) -> dict:
        split_info = filename.split("-")
        return {
            "id": filename,
            "frame": split_info[-1] if len(split_info) < 3 else 0,
            "group": filename if len(split_info) < 3 else "-".join(split_info[:-1])
        }
