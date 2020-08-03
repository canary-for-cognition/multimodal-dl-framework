from preprocessing.classes.base.PipelineComponent import PipelineComponent
from preprocessing.classes.utils.Params import Params


class Preprocessor(PipelineComponent):

    def __init__(self, preprocessor_type: str):
        super().__init__()
        self._params = Params.load_preprocessor_params(preprocessor_type)
