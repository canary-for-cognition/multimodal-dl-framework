import os

import pandas as pd
from tqdm import tqdm


def main():
    path_to_tsv = os.path.join("dataset", "alzheimer", "raw", "eye_tracking_tsv")
    path_to_csv = os.path.join("dataset", "alzheimer", "raw", "eye_tracking_csv")
    os.makedirs(path_to_csv, exist_ok=True)

    print("\n *** TSV 2 CSV *** \n")
    print(" Path to TSV: {}".format(path_to_tsv))
    print(" Path to CSV: {}\n".format(path_to_csv))

    for file_name in tqdm(os.listdir(path_to_tsv)):
        path_to_file = os.path.join(path_to_tsv, file_name)
        pid = file_name.split("_")[1].split(".")[0]
        tsv_file = pd.read_csv(path_to_file, sep="\t", low_memory=False)
        tsv_file.to_csv(os.path.join(path_to_csv, pid + ".csv"), index=False)

    print("\n Conversion completed! \n")


if __name__ == '__main__':
    main()
