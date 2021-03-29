import os

import pandas as pd
from tqdm import tqdm


def main():
    # Either "CookieTheft", "Reading", "Memory" or "PupilCalib"
    task_code = "Reading"

    # Source path to per-participant eye-tracking data
    path_to_raw = os.path.join("raw", "eye_tracking_csv")

    # Destination path to task
    path_to_task = os.path.join("tasks", "reading", "modalities", "raw", "seq_eye_tracking")

    # The experimental log providing the participant type (i.e., the ground truth)
    path_to_participants_log = os.path.join("metadata", "participants_log.tsv")

    # The timestamps defining the tasks for the eye-tracking data
    path_to_timestamps = os.path.join("metadata", "tasks_timestamps.csv")

    # Patient-type-to-label mapping
    labels_map = {"Patient": "1_alzheimer", "Healthy Control (<50)": "0_healthy", "Healthy Control (>50)": "0_healthy"}

    print("\n *** CSV 2 Task ({}) *** \n".format(task_code))
    print(" Path to CSV: {}".format(path_to_raw))
    print(" Path to task: {}\n".format(path_to_task))

    timestamps = pd.read_csv(path_to_timestamps)
    timestamps = timestamps[timestamps["Task"] == task_code]

    patient_type_map = pd.read_csv(path_to_participants_log, sep="\t", usecols=["Study ID", "Participant type"])

    pids = set(timestamps["StudyID"])

    for file_name in tqdm(os.listdir(path_to_raw)):
        pid = file_name.rstrip(".csv")

        if pid not in pids:
            continue

        patient_type = patient_type_map[patient_type_map["Study ID"] == pid]["Participant type"].values[0]

        if patient_type == "Other":
            continue

        start = timestamps[timestamps["StudyID"] == pid]["timestampIni"].values[0]
        end = timestamps[timestamps["StudyID"] == pid]["timestampEnd"].values[0]
        data = pd.read_csv(os.path.join(path_to_raw, file_name), low_memory=False)
        data = data[data["RecordingTimestamp"].between(start + 1, end - 1)]

        label = labels_map[patient_type]
        path_to_label = os.path.join(path_to_task, label)
        os.makedirs(path_to_label, exist_ok=True)

        data.to_csv(os.path.join(path_to_label, file_name))

    print("\n Conversion completed! \n")


if __name__ == '__main__':
    main()
