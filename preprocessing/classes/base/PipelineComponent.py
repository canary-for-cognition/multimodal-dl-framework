from preprocessing.classes.utils.Paths import Paths


class PipelineComponent:

    def __init__(self):
        self._paths = Paths()

    def run(self, *args: any, **kwargs: any):
        pass
