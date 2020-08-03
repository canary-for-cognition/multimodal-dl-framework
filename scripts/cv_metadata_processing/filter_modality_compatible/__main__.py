import os

import pandas as pd


# def filter_items(data: pd.Series, participants: pd.Series):
#     filtered_items = []
#
#     for items in data.iteritems():
#         items = items[1].split(" ")
#         compatible_items = " ".join([item for item in items if item in participants])
#         filtered_items.append(compatible_items)
#
#     return filtered_items

def filter_items(data: pd.Series, participants_audio: pd.Series, participants_text: pd.Series):
    filtered_items = []

    for items in data.iteritems():
        items = items[1].split(" ")
        compatible_items = " ".join([i for i in items if i in participants_audio and i in participants_text])
        filtered_items.append(compatible_items)

    return filtered_items


def main():
    cv_info_folder = "all_data"
    path_to_cv_info_folder = os.path.join("..", "txt_to_csv", "csv", cv_info_folder)
    paths_to_cv_info = [os.path.join(path_to_cv_info_folder, file) for file in os.listdir(path_to_cv_info_folder)]
    path_to_filtered_data = "filtered"
    os.makedirs(path_to_filtered_data, exist_ok=True)

    modalities = ["audio/networks", "text", "sequences/eye_tracking/raw", "eye_tracking/eye_tracking/heatmaps_tobii/networks"]
    path_to_participants = os.path.join("..", "..", "..", "dataset", "alzheimer", "metadata", "participants")

    # for modality in modalities:
    #     modality = "_".join(modality.split(os.sep))
    #     print("\n Filtering {} compatible participants... \n".format(modality))
    #
    #     path_to_modality_compatible = os.path.join(path_to_participants, modality + ".csv")
    #     participants = pd.read_csv(path_to_modality_compatible)["pid"].values
    #
    #     path_to_filtered_compatible = os.path.join(path_to_filtered_data, modality)
    #     os.makedirs(path_to_filtered_compatible, exist_ok=True)
    #
    #     for path_to_cv_info in sorted(paths_to_cv_info):
    #         print("\n\n Processing CV info at {} \n".format(path_to_cv_info))
    #
    #         data = pd.read_csv(path_to_cv_info)
    #         data["train_pos"] = filter_items(data["train_pos"], participants)
    #         data["train_neg"] = filter_items(data["train_neg"], participants)
    #         data["test_pos"] = filter_items(data["test_pos"], participants)
    #         data["test_neg"] = filter_items(data["test_neg"], participants)
    #
    #         print("\t Writing CSV... \n")
    #         data.to_csv(os.path.join(path_to_filtered_compatible, path_to_cv_info.split(os.sep)[-1]), index=False)
    #
    #     print("\n Filtering finished successfully! \n")
    #     print("\n ............................................... \n")

    modality = "language"
    print("\n Filtering {} compatible participants... \n".format(modality))

    participants_audio = pd.read_csv(os.path.join(path_to_participants, "audio_base.csv"))["pid"].values
    participants_text = pd.read_csv(os.path.join(path_to_participants, "text.csv"))["pid"].values

    path_to_filtered_compatible = os.path.join(path_to_filtered_data, modality)
    os.makedirs(path_to_filtered_compatible, exist_ok=True)

    for path_to_cv_info in sorted(paths_to_cv_info):
        print("\n\n Processing CV info at {} \n".format(path_to_cv_info))

        data = pd.read_csv(path_to_cv_info)
        data["train_pos"] = filter_items(data["train_pos"], participants_audio, participants_text)
        data["train_neg"] = filter_items(data["train_neg"], participants_audio, participants_text)
        data["test_pos"] = filter_items(data["test_pos"], participants_audio, participants_text)
        data["test_neg"] = filter_items(data["test_neg"], participants_audio, participants_text)

        print("\t Writing CSV... \n")
        data.to_csv(os.path.join(path_to_filtered_compatible, path_to_cv_info.split(os.sep)[-1]), index=False)

    print("\n Filtering finished successfully! \n")
    print("\n ............................................... \n")


if __name__ == '__main__':
    main()
