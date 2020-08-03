import os

import pandas as pd
from tqdm import tqdm

from preprocessing.classes.base.PipelineComponent import PipelineComponent


class FileFormatConverter(PipelineComponent):

    def __init__(self):
        super().__init__()

        self.__paths_to_tsv = self._paths.get_paths_to_modality(data_folder="preprocessed",
                                                                modality="sequences",
                                                                data_source="eye_tracking",
                                                                representation="tsv")

        self.__paths_to_csv = self._paths.get_paths_to_modality(data_folder="dataset",
                                                                modality="sequences",
                                                                data_source="eye_tracking",
                                                                data_dimension="raw")

    @staticmethod
    def __convert_files(path_to_tsv_files: str, path_to_csv_files: str):
        tsv_files = os.listdir(path_to_tsv_files)
        for file_name in tqdm(tsv_files, desc="Converting TSV files at {} to CSV".format(path_to_tsv_files)):
            tsv_file = os.path.join(path_to_tsv_files, file_name)
            csv_table = pd.read_table(tsv_file, sep='\t')
            csv_table.to_csv(os.path.join(path_to_csv_files, file_name.rstrip("tsv") + "csv"), index=False)

    def __tsv_to_csv(self):
        print("\n----------------------------------------")
        print("         TSV to CSV conversion")
        print("----------------------------------------\n")

        self.__convert_files(self.__paths_to_tsv["pos"], self.__paths_to_csv["pos"])
        self.__convert_files(self.__paths_to_tsv["neg"], self.__paths_to_csv["neg"])

        print("\n----------------------------------------")
        print("     Finished TSV to CSV conversion!")
        print("----------------------------------------\n")

    def run(self, conversion_type: str):
        conversions_map = {
            "tst_to_csv": self.__tsv_to_csv
        }
        conversions_map[conversion_type]()
