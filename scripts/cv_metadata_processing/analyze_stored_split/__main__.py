import os


def count_overlapping_items(l1: list, l2: list):
    l1, l2 = list(set(l1)), list(set(l2))
    long_list, short_list = (l1, l2) if len(l1) > len(l2) else (l2, l1)
    return sum([1 for i in short_list if i in long_list])


def pkl_to_txt(filename: str):
    return "-".join([filename.split("-")[0], filename.split("-")[1]]) + ".txt"


def print_set_analysis(dataset: list, set_type: str, num_pos: int, num_neg: int):
    print("Number of items in the {set_type} set: {tot} \n\n"
          "\t + Alzheimer ......... : {num_pos} ({pos_percentage:.2f}%) \n"
          "\t - Healthy ........... : {num_neg} ({neg_percentage:.2f}%) \n"
          "\t   Unique items ...... : {unique}".format(set_type=set_type,
                                                       tot=len(dataset),
                                                       num_pos=num_pos,
                                                       num_neg=num_neg,
                                                       pos_percentage=(num_pos / len(dataset)) * 100,
                                                       neg_percentage=(num_neg / len(dataset)) * 100,
                                                       unique=len(list(set(dataset)))))

    print("\n-----------------------------------------\n")


def get_unique_patients_by_label(path_to_data: str) -> list:
    return list(set([item for item in os.listdir(path_to_data)]))


def get_unique_patients_in_set(path_to_data: str, labels: tuple) -> list:
    path_to_negative = os.path.join(path_to_data, labels[0])
    negative_data = get_unique_patients_by_label(path_to_negative)

    path_to_positive = os.path.join(path_to_data, labels[1])
    positive_data = get_unique_patients_by_label(path_to_positive)

    return list(set(negative_data + positive_data))


def analyze_set(path_to_dataset: str, set_type: str, labels: tuple):
    pos_items = os.listdir(os.path.join(path_to_dataset, labels[1]))
    neg_items = os.listdir(os.path.join(path_to_dataset, labels[0]))
    dataset = pos_items + neg_items

    print_set_analysis(dataset,
                       set_type,
                       num_pos=len(pos_items),
                       num_neg=len(neg_items))


def main():
    labels = ("0_healthy", "1_alzheimer")
    path_to_dataset = os.path.join("..", "..", "dataset", "alzheimer")
    path_to_split = os.path.join(path_to_dataset, "split", "folds")

    for fold in sorted(os.listdir(path_to_split)):
        print("\n .................................... \n"
              "       Analysis of fold {}"
              "\n .................................... \n".format(fold))

        for set_type in ["training", "validation", "test"]:
            analyze_set(os.path.join(path_to_split, fold, set_type), set_type, labels)

    print("\n .................................................... \n"
          "       Analysis of leakage in training-validation"
          "\n .................................................... \n")

    for i, fold in enumerate(sorted(os.listdir(path_to_split))):
        path_to_training = os.path.join(path_to_split, fold, "training")
        training_patients = get_unique_patients_in_set(path_to_training, labels)

        path_to_validation = os.path.join(path_to_split, fold, "validation")
        validation_patients = get_unique_patients_in_set(path_to_validation, labels)

        path_to_test = os.path.join(path_to_split, fold, "test")
        test_patients = get_unique_patients_in_set(path_to_test, labels)

        training_validation_overlap = count_overlapping_items(training_patients, validation_patients)
        training_test_overlap = count_overlapping_items(training_patients, test_patients)
        validation_test_overlap = count_overlapping_items(validation_patients, test_patients)

        print("\n Fold {}: \n".format(str(i + 1)))
        print("\t - Patients in training ...... : {}".format(len(training_patients)))
        print("\t - Patients in validation .... : {}".format(len(validation_patients)))
        print("\t - Patients in test .......... : {}".format(len(test_patients)))

        print("\n Overlap: \n")
        print("\t - Training   - validation ... : {}".format(training_validation_overlap))
        print("\t - Training   - test ......... : {}".format(training_test_overlap))
        print("\t - Validation - test ......... : {}".format(validation_test_overlap))

        print("\n........................................................\n")


if __name__ == '__main__':
    main()
