import torch
from PIL import Image
from torchvision.transforms import transforms

from classifier.classes.data.loaders.Loader import Loader


class ImageLoader(Loader):

    def __init__(self, for_submodule: bool = False):
        super().__init__("images", for_submodule)
        self.__num_channels = self._modality_params["num_channels"]
        self.__img_size = (self._modality_params["size"]["width"], self._modality_params["size"]["height"])

    def __get_transformations(self) -> list:
        """
        Creates a list of transformations to be applied to the inputs
        :return: a list of transformations to be applied to the inputs
        """
        return [transforms.Resize(self.__img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads an image data item from the dataset
        :param path_to_input: the path to the data item to be loaded referred to the main modality
        :return: the image data item as a tensor
        """
        image = Image.open(self._get_path_to_item(path_to_input))
        transformations = self.__get_transformations()
        return transforms.Compose(transformations)(image)[0:self.__num_channels, :, :]
