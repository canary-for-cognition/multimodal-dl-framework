from preprocessing.classes.base.PipelineComponent import PipelineComponent
from preprocessing.classes.utils.Params import Params


class Generator(PipelineComponent):

    def __init__(self, generator_type: str):
        super().__init__()
        self._params = Params.load_generator_params(generator_type)
