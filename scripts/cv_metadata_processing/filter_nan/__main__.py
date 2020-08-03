import os

import pandas as pd


def filter_items(data: pd.Series, participants: pd.Series):
    filtered_items = []

    for items in data.iteritems():
        items = items[1].split(" ")
        compatible_items = " ".join([item for item in items if item in participants])
        filtered_items.append(compatible_items)

    return filtered_items


def main():
    dataset_type = "alzheimer"
    modality = "sequences"
    data_source = "eye_tracking"
    representation = "random_sampled_gaze_fixations_dropped_na"

    path_to_dataset = os.path.join("..", "..", "..", "dataset", dataset_type)
    base_path_to_metadata = os.path.join(path_to_dataset, "split", "metadata")
    path_to_data_source = os.path.join(data_source, "10")
    path_to_metadata = os.path.join(base_path_to_metadata, path_to_data_source)
    path_to_filtered = os.path.join("filtered", data_source, modality, representation)
    os.makedirs(path_to_filtered, exist_ok=True)

    participants = pd.read_csv(os.path.join("participants.csv"), usecols=["pid"])["pid"].values

    for file_name in os.listdir(path_to_metadata):
        path_to_cv_info = os.path.join(path_to_metadata, file_name)
        cv_info = pd.read_csv(path_to_cv_info)
        print("\n\n Processing CV info at {} \n".format(path_to_cv_info))

        data = pd.read_csv(path_to_cv_info)
        data["train_pos"] = filter_items(cv_info["train_pos"], participants)
        data["train_neg"] = filter_items(cv_info["train_neg"], participants)
        data["test_pos"] = filter_items(cv_info["test_pos"], participants)
        data["test_neg"] = filter_items(cv_info["test_neg"], participants)

        print("\t Writing CSV... \n")
        data.to_csv(os.path.join(path_to_filtered, path_to_cv_info.split(os.sep)[-1]), index=False)

    print("\n Filtering finished successfully! \n")
    print("\n ............................................... \n")


if __name__ == '__main__':
    main()
