import os

import pandas as pd


def get_unique_items(path_to_items: str) -> list:
    return [file_name.split(".")[0] for file_name in os.listdir(path_to_items)]


def main():
    labels = {
        "negative": "0_healthy",
        "positive": "1_alzheimer"
    }
    base_path_to_modalities = os.path.join("..", "..", "..", "dataset", "alzheimer", "modalities")
    modalities = ["audio/networks", "text", "sequences/eye_tracking/raw", "eye_tracking/eye_tracking/heatmaps_tobii/networks"]
    paths_to_modalities = [os.path.join(base_path_to_modalities, modality) for modality in modalities]
    path_to_participants_lists = "participants_lists"

    print("\n .................................... \n"
          "       Analysis of compatible items"
          "\n .................................... \n")

    for path_to_modality in paths_to_modalities:
        path_to_negative = os.path.join(path_to_modality, labels["negative"])
        path_to_positive = os.path.join(path_to_modality, labels["positive"])
        negative_items = get_unique_items(path_to_negative)
        positive_items = get_unique_items(path_to_positive)
        items = sorted(negative_items + positive_items)
        data = pd.DataFrame({"pid": items})

        modality = "_".join(path_to_modality.split(os.sep)[6:])
        print("\n {} participants: \n".format(modality))
        print(data)

        list_file_name = modality + ".csv"
        data.to_csv(os.path.join(path_to_participants_lists, list_file_name))
        print("\n {} saved! \n".format(list_file_name))
        print("\n .................................... \n")


if __name__ == '__main__':
    main()
