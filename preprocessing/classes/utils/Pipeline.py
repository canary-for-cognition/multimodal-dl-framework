from preprocessing.classes.generators.images.HSHGenerator import HSHGenerator
from preprocessing.classes.generators.images.HeatmapsGenerator import HeatmapsGenerator
from preprocessing.classes.generators.images.MFCCGenerator import MFCCGenerator
from preprocessing.classes.generators.images.MelSpectrogramsGenerator import MelSpectrogramsGenerator
from preprocessing.classes.generators.images.ScanPathsGenerator import ScanPathsGenerator
from preprocessing.classes.generators.sequences.AudioSequencesGenerator import AudioSequencesGenerator
from preprocessing.classes.preprocessors.sequences.AudioSequencesPreprocessor import AudioSequencesPreprocessor
from preprocessing.classes.preprocessors.sequences.EyeTrackingSequencesPreprocessor import \
    EyeTrackingSequencesPreprocessor
from preprocessing.classes.preprocessors.text.TextPreprocessor import TextPreprocessor


class Pipeline:

    def __init__(self):
        self.__pipeline_map = {
            # "filter_by_label": FileFilter().run(discriminant="label"),
            # "filter_by_task_cookie_theft": TaskFilter().run(task_code="cookie_theft"),
            # "convert_tsv_to_csv": FileFormatConverter().run(conversion_type="tsv_to_csv"),
            "preprocess_eye_tracking_sequences": EyeTrackingSequencesPreprocessor(),
            "preprocess_audio_sequences": AudioSequencesPreprocessor(),
            "preprocess_text": TextPreprocessor(),
            "generate_heatmaps": HeatmapsGenerator(),
            "generate_hsh": HSHGenerator(),
            "generate_scan_paths": ScanPathsGenerator(),
            "generate_mfcc": MFCCGenerator(),
            "generate_mel_spectrograms": MelSpectrogramsGenerator(),
            "generate_audio_sequences": AudioSequencesGenerator()
        }

    @staticmethod
    def __print_heading(index: int, operation: str):
        print("\n--------------------------------------------------------")
        print("     Step {} - {}".format(index, operation))
        print("--------------------------------------------------------\n")

    @staticmethod
    def __print_footer(index: int, operation: str):
        print("\n--------------------------------------------------------")
        print("     Finished step {} - {}!".format(index, operation))
        print("--------------------------------------------------------\n\n")
        print("########################################################\n")

    def run(self, operations: list):
        print("\n Running preprocessing pipeline with the following operations:\n")
        for i, operation in enumerate(operations):
            print("\t - {}) {}".format(i + 1, operation))
        print("\n ................................. \n")

        if input("Confirm operations? [y/n]") == "y":
            for i, operation in enumerate(operations):
                self.__print_heading(i + 1, operation)
                self.__pipeline_map[operation].run()
                self.__print_footer(i + 1, operation)
        else:
            print("Pipeline aborted! No operation on data has been performed.")
