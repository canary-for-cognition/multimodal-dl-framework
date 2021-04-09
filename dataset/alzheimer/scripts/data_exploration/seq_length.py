import os
import pickle
from statistics import median, mean

import matplotlib.pyplot as plt


def main():
    path_to_dataset = os.path.join("dataset", "alzheimer", "tasks", "reading", "modalities", "preprocessed")
    path_to_seq = os.path.join(path_to_dataset, "sequences", "eye_tracking_coll_fix", "augmented")
    path_to_pos, path_to_neg = os.path.join(path_to_seq, "1_alzheimer"), os.path.join(path_to_seq, "0_healthy")
    pos_seq = sorted([os.path.join(path_to_pos, seq) for seq in os.listdir(path_to_pos)])
    neg_seq = sorted([os.path.join(path_to_neg, seq) for seq in os.listdir(path_to_neg)])

    seq_lengths = [pickle.load(open(seq, "rb")).shape[0] for seq in pos_seq + neg_seq]

    print("\n Max: {} - Avg: {:.2f} - Median: {:.2f} \n"
          .format(max(seq_lengths), mean(seq_lengths), median(seq_lengths)))

    num_items = len(seq_lengths)
    avg_length = mean(seq_lengths)

    plt.plot(range(num_items), seq_lengths, label="ET coll fix")
    plt.plot(range(num_items), [avg_length] * num_items, label="Avg")
    plt.legend()
    plt.title("Avg len: ~{}".format(int(avg_length)))
    plt.xlabel("Sequence")
    plt.ylabel("Length")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
