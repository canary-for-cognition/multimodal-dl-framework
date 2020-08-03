import os
import shutil

from tqdm import tqdm

from preprocessing.classes.base.PipelineComponent import PipelineComponent


class FileFilter(PipelineComponent):

    def __init__(self):
        super().__init__()

        self.__path_to_raw = self._paths.get_paths_to_modality(data_folder="dataset",
                                                               modality="sequences",
                                                               data_source="eye_tracking",
                                                               data_dimension="raw",
                                                               return_base_path=True)

        self.__path_to_tsv = self._paths.get_paths_to_modality(data_folder="preprocessed",
                                                               modality="sequences",
                                                               data_source="eye_tracking",
                                                               representation="tsv")

    def __filter_by_label(self):
        for file_name in tqdm(os.listdir(self.__path_to_raw), desc="Filtering raw files by labels"):
            path_to_file = os.path.join(self.__path_to_raw, file_name)
            clean_file_name = file_name.split("_")[1]
            path_to_preprocessed_file = os.path.join(self.__path_to_tsv[clean_file_name[0]], clean_file_name)
            shutil.copy(path_to_file, path_to_preprocessed_file)

    def run(self, discriminant: str):
        filters_map = {
            "label": self.__filter_by_label
        }
        filters_map[discriminant]()
