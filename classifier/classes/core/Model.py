from typing import Union

import torch

from classifier.classes.factories.CriterionFactory import CriterionFactory
from classifier.classes.factories.OptimizerFactory import OptimizerFactory


class Model:

    def __init__(self, device: torch.device):
        self._device = device
        self._network, self.__optimizer, self.__criterion = None, None, None

    def predict(self, x: Union[torch.Tensor, dict], *args: any, **kwargs: any) -> Union[tuple, torch.Tensor]:
        """
        Performs a prediction using the network and returns the output logits
        """
        return self._network(x.to(self._device))

    def print_model_overview(self):
        """
        Prints the architecture of the network
        """
        print("\n Model overview: \n")
        print(self._network)

    def train_mode(self):
        """
        Sets the network to train mode
        """
        self._network = self._network.train()

    def evaluation_mode(self):
        """
        Sets the network to evaluation mode (i.e. batch normalization and dropout layers will work
        in evaluation mode instead of train mode)
        """
        self._network = self._network.eval()

    def get_loss(self, o: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes the loss for the given logits and ground truth
        :param o: the logits
        :param y: the ground truth
        :return: the loss value based on the set criterion
        """
        return self.__criterion(o, y).item()

    def reset_gradient(self):
        """ Zeros out all the accumulated gradients """
        self.__optimizer.zero_grad()

    def update_weights(self, o: torch.Tensor, y: torch.Tensor) -> float:
        """
        Updates the weights of the model performing backpropagation
        :param o: the output of the forward step
        :param y: the ground truth
        :return: the loss value for the current update of the weights
        """
        loss = self.__criterion(o, y)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    def set_optimizer(self, optimizer_type: str, learning_rate: float):
        """
        Instantiates the optimizer for the train
        :param optimizer_type: the type of optimizer to be instantiated, in {Adam, SGD}
        :param learning_rate: the initial learning rate
        :return: an optimizer in {Adam, SGD}
        """
        print("\n Optimizer: {} (learning rate is {})".format(optimizer_type, learning_rate))
        self.__optimizer = OptimizerFactory(list(self._network.parameters()), learning_rate).get(optimizer_type)

    def set_criterion(self, criterion_type: str):
        """
        Instantiates a criterion for the train
        :param criterion_type: the type of criterion to be instantiated, in {NLLLoss, CrossEntropyLoss}
        :return: a criterion in {NLLLoss, CrossEntropyLoss}
        """
        print("\n Criterion: {}".format(criterion_type))
        self.__criterion = CriterionFactory().get(criterion_type).to(self._device)

    def save(self, path_to_model: str):
        """
        Saves the current model at the given path
        :param path_to_model: the path where to save the current model at
        """
        torch.save(self._network.state_dict(), path_to_model)

    def load(self, path_to_model: str):
        """
        Loads a model from the given path
        :param path_to_model: the path where to load the model from
        """
        print("\n Loading model... \n")
        self._network.load_state_dict(torch.load(path_to_model))
