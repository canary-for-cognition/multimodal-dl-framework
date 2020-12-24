class AlzheimerGroupingPolicy:

    @staticmethod
    def group(filename: str) -> str:
        split_info = filename.split("-")
        return filename if len(split_info) < 3 else "-".join(split_info[:-1])
