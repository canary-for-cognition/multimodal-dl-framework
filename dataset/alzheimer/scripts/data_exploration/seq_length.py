import os
import pickle

import matplotlib.pyplot as plt


def main():
    path_to_dataset = os.path.join("dataset", "alzheimer", "tasks", "reading", "modalities", "preprocessed")
    path_to_seq = os.path.join(path_to_dataset, "sequences", "eye_tracking", "augmented")
    path_to_pos, path_to_neg = os.path.join(path_to_seq, "1_alzheimer"), os.path.join(path_to_seq, "0_healthy")
    pos_seq = [os.path.join(path_to_pos, seq) for seq in os.listdir(path_to_pos)]
    neg_seq = [os.path.join(path_to_neg, seq) for seq in os.listdir(path_to_neg)]

    seq_lengths = [len(pickle.load(open(seq, "rb"))) for seq in pos_seq + neg_seq]

    num_items = len(seq_lengths)
    avg_length = sum(seq_lengths) / num_items
    print("\n Average sequence length: {} \n".format(avg_length))

    plt.plot(range(num_items), seq_lengths)
    plt.plot(range(num_items), [avg_length] * num_items, label="Average")
    plt.legend()
    plt.title("Avg len: ~{}".format(int(avg_length)))
    plt.xlabel("Sequence")
    plt.ylabel("Length")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
