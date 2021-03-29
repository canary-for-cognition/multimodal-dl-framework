import os

import pandas as pd


def analyze_folds(folds: pd.DataFrame, dirty_subset: pd.DataFrame):
    unique_items_in_file = []

    for i, (_, row) in enumerate(folds.iterrows()):
        train_pos, train_neg = row["train_pos"].split(" "), row["train_neg"].split(" ")
        test_pos, test_neg = row["test_pos"].split(" "), row["test_neg"].split(" ")

        unique_items_in_file += train_pos + train_neg + test_pos + test_neg

        train_dirty_pos = [item for item in train_pos if item in list(dirty_subset["PID"])]
        train_dirty_neg = [item for item in train_neg if item in list(dirty_subset["PID"])]
        train_dirty = train_dirty_pos + train_dirty_neg

        test_dirty_pos = [item for item in test_pos if item in list(dirty_subset["PID"])]
        test_dirty_neg = [item for item in test_neg if item in list(dirty_subset["PID"])]
        test_dirty = test_dirty_pos + test_dirty_neg

        print("\n Fold {n}: \n\n"
              "\t + Alzheimer in training ... : {num_pos_train} \n"
              "\t - Healthy in training ..... : {num_neg_train} \n"
              "\t § Dirty training data ..... : {dirty_training} \n\n"
              "\t + Alzheimer in test ....... : {num_pos_test} \n"
              "\t - Healthy in test ......... : {num_neg_test} \n"
              "\t § Dirty test data ......... : {dirty_test} \n".format(n=i + 1,
                                                                        num_pos_train=len(train_pos),
                                                                        num_neg_train=len(train_neg),
                                                                        num_pos_test=len(test_pos),
                                                                        num_neg_test=len(test_neg),
                                                                        dirty_training=len(train_dirty),
                                                                        dirty_test=len(test_dirty)))

        print("------------------------------------------------------------")

    unique_items_in_file = list(set(unique_items_in_file))
    dirty_items_in_file = [item for item in unique_items_in_file if item in list(dirty_subset["PID"])]
    print("\n Unique items in file: {} | Dirty items in file: {} \n".format(len(unique_items_in_file),
                                                                            len(dirty_items_in_file)))

    return unique_items_in_file


def convert_txt_to_csv(file_base_name: str) -> pd.DataFrame:
    print("\nConverting {f}.txt to {f}.csv...\n".format(f=file_base_name))

    train_test_split = {"train": [], "test": []}
    class_split = {"train": {"pos": [], "neg": []}, "test": {"pos": [], "neg": []}}

    for line in open(os.path.join("txt", file_base_name) + ".txt", "r").readlines():
        split_line = line.split(", ")
        data_type = split_line[0][0:-1].lower()
        train_test_split[data_type] += [split_line[1:-1]]

    for item in train_test_split["train"]:
        class_split["train"]["neg"] += [" ".join([i.strip("'") for i in item if i[1] == "neg"])]
        class_split["train"]["pos"] += [" ".join([i.strip("'") for i in item if i[1] == "pos"])]

    for item in train_test_split["test"]:
        class_split["test"]["neg"] += [" ".join([i.strip("'") for i in item if i[1] == "neg"])]
        class_split["test"]["pos"] += [" ".join([i.strip("'") for i in item if i[1] == "pos"])]

    data = {
        "train_pos": class_split["train"]["pos"],
        "train_neg": class_split["train"]["neg"],
        "test_pos": class_split["test"]["pos"],
        "test_neg": class_split["test"]["neg"]
    }
    folds = pd.DataFrame(data)

    print("Writing CSV file...\n")

    folds.to_csv(os.path.join("csv", file_base_name) + ".csv", index=False)

    print("Conversion completed!\n")

    return folds


def convert_files(file_base_namesList, dataset: pd.DataFrame) -> List:
    unique_items = []
    for file_base_name in file_base_names:
        folds = convert_txt_to_csv(file_base_name)
        print("------------------------------------------------------------")
        unique_items += [analyze_folds(folds, dataset[dataset["DataClean"] == "NO"])]
        print("############################################################")
    return unique_items


def print_dataset_overview(dataset: pd.DataFrame, dataset_name: str):
    print("The {dt} counts {n} items:\n"
          "\t + Alzheimer ... : {num_pos}\n"
          "\t - Healthy ..... : {num_neg}\n"
          "\t § Dirty ....... : {dirty}\n".format(dt=dataset_name,
                                                  n=dataset.shape[0],
                                                  num_pos=len(dataset[dataset["Label"] == "pos"]),
                                                  num_neg=len(dataset[dataset["Label"] == "neg"]),
                                                  dirty=len(dataset[dataset["DataClean"] == "NO"])))


def check_patients_availability(unique_itemsList, path_to_sequences: str, labels: Dict):
    for item in list(set([item for items in unique_items for item in items])):
        file_name = item + "-1.pkl"
        if not os.path.exists(os.path.join(path_to_sequences, labels[item[0]], file_name)):
            print("WARNING: patient {} is not present within the data!".format(item))


def check_sets_difference(unique_itemsList, dataset: pd.DataFrame):
    difference = list(set(unique_items[1]) - set(unique_items[0]))
    dirty_difference = [item for item in difference if item in list(dataset[dataset["DataClean"] == "NO"]["PID"])]
    print("\n Difference between the full data and the subset: \n"
          "\t - Number of elements ......... : {} \n"
          "\t - Number of dirty elements ... : {} \n".format(len(difference), len(dirty_difference)))


def main():
    labels = {
        "pos": "1_alzheimer",
        "neg": "0_healthy"
    }
    path_to_dataset = os.path.join("", "../../../../utils", "..", "..")
    path_to_sequences = os.path.join(path_to_dataset, "modalities", "eye_tracking", "sequences")
    path_to_participants = os.path.join(path_to_dataset, "metadata", "participants_info.csv")
    columns = ["PID", "ParticipantType", "DataClean", "RestPupil", "Label"]
    dataset = pd.read_csv(path_to_participants, usecols=columns, index_col=False)

    print_dataset_overview(dataset=dataset, dataset_name="full data")
    print_dataset_overview(dataset=dataset[dataset["DataClean"] == "YES"], dataset_name="clean subset")
    print_dataset_overview(dataset=dataset[dataset["DataClean"] == "NO"], dataset_name="dirty subset")

    print("\n------------------------------------------------------------")

    # file_base_names = ["cv_folds_subset", "cv_folds_superset"]

    cv_info_folder = "all_data"
    path_to_cv_info_folder = os.path.join("txt", cv_info_folder)
    file_base_names = [os.path.join(cv_info_folder, file_name[:-4]) for file_name in os.listdir(path_to_cv_info_folder)]
    unique_items = convert_files(file_base_names, dataset)

    check_patients_availability(unique_items, path_to_sequences, labels)
    check_sets_difference(unique_items, dataset)


if __name__ == '__main__':
    main()
