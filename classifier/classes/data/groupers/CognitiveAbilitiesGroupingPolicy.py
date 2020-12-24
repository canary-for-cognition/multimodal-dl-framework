class CognitiveAbilitiesGroupingPolicy:

    @staticmethod
    def group(filename: str) -> str:
        return filename.split(".")[0].split("-")[0]
