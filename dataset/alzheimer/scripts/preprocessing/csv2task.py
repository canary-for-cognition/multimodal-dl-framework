import os

import pandas as pd
from tqdm import tqdm


def main():
    # Either "CookieTheft", "Reading", "Memory" or "PupilCalib"
    task_code = "Reading"

    # The list of participant ids (PIDs) involved in the selected task
    path_to_participants = os.path.join("tasks", "reading", "metadata", "participants.csv")

    # The experimental log providing the participant type (i.e., the ground truth)
    path_to_participants_log = os.path.join("metadata", "participants_log.tsv")

    # Source path to per-participant eye-tracking data
    path_to_raw = os.path.join("raw", "eye_tracking_csv")

    # Destination path to task
    path_to_task = os.path.join("tasks", "reading", "modalities", "raw", "seq_eye_tracking")

    # The timestamps defining the tasks for the eye-tracking data
    path_to_timestamps = os.path.join("metadata", "tasks_timestamps.csv")

    # Patient-type-to-label mapping
    labels_map = {"Patient": "1_alzheimer", "Healthy Control (>50)": "0_healthy"}

    print("\n *** CSV 2 Task ({}) *** \n".format(task_code))
    print(" Path to CSV .... : {}".format(path_to_raw))
    print(" Path to task ... : {}\n".format(path_to_task))

    timestamps = pd.read_csv(path_to_timestamps)
    timestamps = timestamps[timestamps["Task"] == task_code]

    task_pids = set(pd.read_csv(path_to_participants)["PID"].to_list())
    participants_data = pd.read_csv(path_to_participants_log, sep="\t",
                                    usecols=["Study ID", "Participant type", "Eye-Tracking Calibration?"])

    all_pids = set(timestamps["StudyID"])

    valid_pids = []

    for file_name in tqdm(os.listdir(path_to_raw)):
        pid = file_name.rstrip(".csv")

        if pid not in task_pids:
            print("\n WARNING: PID {} not present in task PIDs. Skipping \n".format(pid))
            continue

        if pid not in all_pids:
            print("\n WARNING: PID {} not present in experiment PIDs. Skipping \n".format(pid))
            continue

        patient_type = participants_data[participants_data["Study ID"] == pid]["Participant type"].values[0]

        if patient_type in ["Other", "Healthy Control (<50)"]:
            print("\n WARNING: PID {} is '{}'. Skipping \n".format(pid, patient_type))
            continue

        calibration = participants_data[participants_data["Study ID"] == pid]["Eye-Tracking Calibration?"].values[0]

        if not calibration:
            print("\n WARNING: PID {} has issues with calibration. Skipping \n".format(pid))
            continue

        print("\n -> PID {} is valid! \n".format(pid))
        valid_pids.append(pid)

        start = timestamps[timestamps["StudyID"] == pid]["timestampIni"].values[0]
        end = timestamps[timestamps["StudyID"] == pid]["timestampEnd"].values[0]
        data = pd.read_csv(os.path.join(path_to_raw, file_name), low_memory=False)
        data = data[data["RecordingTimestamp"].between(start + 1, end - 1)]

        label = labels_map[patient_type]
        path_to_label = os.path.join(path_to_task, label)
        os.makedirs(path_to_label, exist_ok=True)

        data.to_csv(os.path.join(path_to_label, file_name))

    print("\n Conversion completed for {} valid PIDs! \n".format(len(valid_pids)))


if __name__ == '__main__':
    main()
