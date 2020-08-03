import json
import os

import pandas as pd


def main():
    target_experiment = "5_iterations_10_folds_vtnet_blank_heatmaps_base_1"
    path_to_experiment = os.path.join("..", "..", "classifier", "experiments", "highlights", target_experiment)
    path_to_aggregated_results = os.path.join("aggregated_results", target_experiment)
    os.makedirs(path_to_aggregated_results, exist_ok=True)

    seeds = [item for item in os.listdir(path_to_experiment) if "seed" in item]
    for seed in seeds:
        iterations_data = []
        iterations = os.listdir(os.path.join(path_to_experiment, seed))
        for iteration in iterations:
            path_to_metrics = os.path.join(path_to_experiment, seed, iteration, "metrics")
            iterations_data += [json.load(open(os.path.join(path_to_metrics, "cv_average.json")))["test"]]
        df = pd.DataFrame(iterations_data, columns=iterations_data[0].keys())
        file_name = seed + "_with_" + str(len(iterations)) + "_iterations.csv"
        df.to_csv(os.path.join(path_to_aggregated_results, file_name))
        print("\n Seed # {} \n".format(seed.split("_")[1]))
        print(df.head(10))


if __name__ == '__main__':
    main()
