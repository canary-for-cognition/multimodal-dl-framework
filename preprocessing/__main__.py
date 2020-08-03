from preprocessing.classes.utils.Pipeline import Pipeline


def main():
    """
    Supported pipeline operations:
        - "filter_by_label": split the files by label into different folders
        - "convert_tsv_to_csv": converts the data item from tsv to csv
        - "filter_by_task_cookie_theft": filters out the rows which are not related to the Cookie Theft task
           in the raw data
        - "preprocess_audio_sequences": preprocesses the audio sequences
        - "preprocess_eye_tracking_sequences": preprocesses the eye-tracking sequences
        - "preprocess_text": preprocesses the text
        - "generate_heatmaps": generates the heatmaps from the eye-tracking sequences
        - "generate_hsh": generates the hsh from the eye-tracking sequences
        - "generate_scan_paths": generates the scan-paths from the eye-tracking sequences
        - "generate_mfcc_spectrograms": generates the MFCC spectrogram as a visual representation of audio
        - "generate_audio_sequences": generates the audio sequences from the raw audio WAV files
    """
    operations = ["generate_heatmaps"]
    Pipeline().run(operations)


if __name__ == '__main__':
    main()
