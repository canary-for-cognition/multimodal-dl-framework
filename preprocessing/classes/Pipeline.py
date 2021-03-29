from typing import List

from preprocessing.classes.generators.images.HSHGenerator import HSHGenerator
from preprocessing.classes.generators.images.HeatmapsGenerator import HeatmapsGenerator
from preprocessing.classes.generators.images.MFCCGenerator import MFCCGenerator
from preprocessing.classes.generators.images.MelSpectrogramsGenerator import MelSpectrogramsGenerator
from preprocessing.classes.generators.images.ScanPathsGenerator import ScanPathsGenerator
from preprocessing.classes.generators.sequences.AudioSeqGenerator import AudioSeqGenerator
from preprocessing.classes.preprocessors.sequences.AudioSeqPreprocessor import AudioSeqPreprocessor
from preprocessing.classes.preprocessors.sequences.ETSeqPreprocessor import ETSeqPreprocessor
from preprocessing.classes.preprocessors.text.TextPreprocessor import TextPreprocessor


class Pipeline:

    def __init__(self):
        self.__pipeline_map = {
            "preprocess_eye_tracking_sequences": ETSeqPreprocessor,
            "preprocess_audio_sequences": AudioSeqPreprocessor,
            "preprocess_text": TextPreprocessor,
            "generate_heatmaps": HeatmapsGenerator,
            "generate_hsh": HSHGenerator,
            "generate_scan_paths": ScanPathsGenerator,
            "generate_mfcc": MFCCGenerator,
            "generate_mel_spectrograms": MelSpectrogramsGenerator,
            "generate_audio_sequences": AudioSeqGenerator
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

    def run(self, operations: List):
        print("\n Running preprocessing pipeline with the following operations:\n")
        for i, operation in enumerate(operations):
            print("\t - {}) {}".format(i + 1, operation))
        print("\n ................................. \n")

        if input("Confirm operations? [y/n]") == "y":
            for i, operation in enumerate(operations):
                self.__print_heading(i + 1, operation)
                self.__pipeline_map[operation]().run()
                self.__print_footer(i + 1, operation)
        else:
            print("Pipeline aborted! No operation on data has been performed.")
