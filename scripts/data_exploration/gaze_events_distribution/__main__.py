import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def add_events_indices(sequences: pd.DataFrame):
    events = sequences["GazeEventType"].values
    counters = {e: 0 for e in events}
    indices = []
    for i, e in enumerate(events):
        indices += [counters[e]]
        if i + 1 < len(events) and e != events[i + 1]:
            counters[e] += 1

    for i, e in zip(indices, events):
        print(i, e)


def group_by_event(path_to_items: str) -> pd.DataFrame:
    grouped_sequences = []
    for filename in tqdm(os.listdir(path_to_items), desc="Grouping sequences by event at {}".format(path_to_items)):
        sequences = pd.read_csv(os.path.join(path_to_items, filename))
        add_events_indices(sequences)
        exit()
        sequences = sequences[sequences["GazeEventType"] == "Saccade"][["GazePointX (ADCSpx)", "GazePointY (ADCSpx)"]]
        print(sequences.head())
        sequences = sequences.dropna().agg("mean", axis="columns")
        print(sequences.head())
        plt.scatter(sequences["GazePointX (ADCSpx)"], sequences["GazePointY (ADCSpx)"], c="b", alpha=0.5)
        plt.plot(sequences["GazePointX (ADCSpx)"], sequences["GazePointY (ADCSpx)"], c="k", alpha=0.25)
        plt.scatter(sequences["GazePointX (ADCSpx)"].mean(), sequences["GazePointY (ADCSpx)"].mean(), c="r")
        plt.title(filename)
        plt.show()
        exit()
        sequences = sequences.groupby("GazeEventType").size()
        grouped_sequences += [sequences.reset_index().rename(columns={0: 'NumEvents'})]
    return pd.concat(grouped_sequences)


def main():
    dataset_name = "alzheimer"
    data_source = "eye_tracking"
    labels = {
        "pos": "1_alzheimer",
        "neg": "0_healthy",
    }

    path_to_modalities = os.path.join("..", "..", "..", "dataset", dataset_name, "modalities")
    path_to_sequences = os.path.join(path_to_modalities, "sequences", data_source, "raw")
    paths_to_sequences = {
        "pos": os.path.join(path_to_sequences, labels["pos"]),
        "neg": os.path.join(path_to_sequences, labels["neg"])
    }

    items = pd.concat([group_by_event(paths_to_sequences["pos"]), group_by_event(paths_to_sequences["neg"])])
    for event_type in set(items["GazeEventType"].values):
        event_counts = items[items["GazeEventType"] == event_type]["NumEvents"].values
        print("\n Avg number of {} events: {} \n".format(event_type.lower(), event_counts.mean()))
        plt.plot(range(len(event_counts)), event_counts, label=event_type)

    plt.legend()
    plt.xlabel("Num events")
    plt.ylabel("Data items")
    plt.show()


if __name__ == '__main__':
    main()
