import os
import pickle

import matplotlib.pyplot as plt


def get_lengths(path_to_sequences: str) -> list:
    lengths = []
    for filename in os.listdir(path_to_sequences):
        sequence = pickle.load(open(os.path.join(path_to_sequences, filename), "rb"))
        # print(sequence.columns)
        # exit()
        # sequences = sequences.drop_duplicates(subset=["FixationPointX (MCSpx)", "FixationPointY (MCSpx)"], keep="first")
        lengths += [len(sequence)]
    return lengths


def main():
    data_dimension = "base"
    representation = "collapsed_fixations"

    file_name = "et_sequences_lengths_" + representation + "_" + data_dimension
    title = "Lengths of sequences using \n" + " ".join(representation.split("_")) + " (" + data_dimension + ")"
    x_label, y_label = "Sequence", "Length"
    path_to_dataset = os.path.join("..", "..", "..", "dataset", "alzheimer", "modalities")
    path_to_sequences = os.path.join(path_to_dataset, "sequences", "eye_tracking", representation, data_dimension)
    path_to_sequences_pos = os.path.join(path_to_sequences, "1_alzheimer")
    path_to_sequences_neg = os.path.join(path_to_sequences, "0_healthy")

    lengths = get_lengths(path_to_sequences_pos) + get_lengths(path_to_sequences_neg)

    num_items = len(lengths)
    avg_length = sum(lengths) / num_items
    plt.plot(range(num_items), lengths)
    plt.plot(range(num_items), [avg_length] * num_items, label="Average")
    plt.legend()
    plt.title(title + "\n AVG: ~{}\n".format(int(avg_length)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join("..", "..", "..", "docs", file_name + ".png"))
    plt.show()
    plt.clf()

    print("\n Average sequence length: {} \n".format(avg_length))


if __name__ == '__main__':
    main()
