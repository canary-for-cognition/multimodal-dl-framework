import os

import pandas as pd
from tqdm import tqdm

from preprocessing.classes.base.PipelineComponent import PipelineComponent


class TaskFilter(PipelineComponent):

    def __init__(self):
        super().__init__()

        self.__paths_to_csv = self._paths.get_paths_to_modality(data_folder="dataset",
                                                                modality="sequences",
                                                                data_source="eye_tracking",
                                                                data_dimension="raw")

        self.__path_to_timestamps = self._paths.get_metadata(metadata_type="timestamps")

        self.__paths_to_tasks = self._.get_paths_to_tasks()
        self.__tasks_map = {
            "cookie_theft": "CookieTheft",
            "pupil_calibration": "PupilCalib",
            "reading": "Reading",
            "memory": "Memory"
        }

    @staticmethod
    def __filter_task_rows(path_to_raw: str, path_to_task: str, timestamps: pd.DataFrame):
        for file_name in tqdm(os.listdir(path_to_raw), desc="Filtering tasks at {}".format(path_to_raw)):
            pid = file_name.rstrip(".csv")
            start = timestamps[timestamps["StudyID"] == pid]["timestampIni"].values[0]
            end = timestamps[timestamps["StudyID"] == pid]["timestampEnd"].values[0]
            data = pd.read_csv(os.path.join(path_to_raw, file_name))
            data = data[data["RecordingTimestamp"].between(start + 1, end - 1)]
            data.to_csv(os.path.join(path_to_task, file_name))

    def __filter_task(self, task_code: str):
        task_type = self.__tasks_map[task_code]
        timestamps = pd.read_csv(self.__path_to_timestamps)
        timestamps = timestamps[timestamps["Task"] == task_type]
        self.__filter_task_rows(self.__paths_to_csv["pos"], self.__paths_to_tasks[task_code]["pos"], timestamps)
        self.__filter_task_rows(self.__paths_to_csv["neg"], self.__paths_to_tasks[task_code]["neg"], timestamps)

    def run(self, task_code: str):
        self.__filter_task(task_code)
