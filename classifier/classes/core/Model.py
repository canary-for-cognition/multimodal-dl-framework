from typing import Iterator, Union

import torch
from torch import nn

from classifier.classes.factories.CriterionFactory import CriterionFactory
from classifier.classes.factories.OptimizerFactory import OptimizerFactory


class Model:
    def __init__(self, device: torch.device):
        self._device = device
        self._network, self.__optimizer, self.__criterion, self.__clip_gradient = None, None, None, None

    def predict(self, inputs: Union[torch.Tensor, dict], *args: any, **kwargs: any) -> Union[tuple, torch.Tensor]:
        """
        Performs a prediction using the network and returns the output logits
        """
        return self._network(inputs.float().to(self._device))

    def get_init_state(self, batch_size: int) -> torch.Tensor:
        """
        Returns the initial state of the network (for stateful models only)
        :param batch_size: the size of the batch
        :return: the initial state of the network
        """
        return self._network.init_state(batch_size).to(self._device)

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
        in evaluation mode instead of training mode)
        """
        self._network = self._network.eval()

    def get_named_parameters(self) -> Iterator:
        """
        Returns an iteration over the parameters of the network, yielding both the name of the parameter as well as
        the parameter itself
        :return: an iteration over the parameters of the network, yielding both the name of the parameter as well as
        the parameter itself
        """
        return self._network.named_parameters()

    def get_parameters(self) -> list:
        """
        Returns the parameters of the network in a list
        :return: the parameters of the network in a list
        """
        return list(self._network.parameters())

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Computes the loss for the given logits and ground truth
        :param outputs: the logits
        :param labels: the ground truth
        :return: the loss value based on the active criterion
        """
        return self.__criterion(outputs, labels).item()

    def optimize(self):
        """
        Performs the optimization step
        """
        self.__optimizer.step()

    def update_weights(self,
                       outputs: torch.Tensor,
                       labels: torch.Tensor,
                       retain_graph: bool = True,
                       optimize: bool = True) -> float:
        """
        Updates the weights of the model performing standard backpropagation (with gradient clipping, if enabled)
        :param outputs: the output of the forward step
        :param labels: the ground truth
        :param retain_graph: whether or not to retain the computational graph when performing the backward step
        :param optimize: whether or not to perform the optimization step
        :return: the loss value for the current update of the weights
        """
        loss = self.__criterion(outputs, labels)
        self.__optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if self.__clip_gradient:
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=0.2)

        if optimize:
            self.optimize()

        return loss.item()

    def set_optimizer(self, optimizer_type: str, learning_rate: float, clip_gradient: bool):
        """
        Instantiates the optimizer for the training
        :param optimizer_type: the type of optimizer to be instantiated, in {Adam, SGD}
        :param learning_rate: the initial learning rate
        :param clip_gradient: whether or not to clip the gradient to prevent gradient explosions
        :return: an optimizer in {Adam, SGD}
        """
        print("\n Optimizer: {} (initial learning rate is {}, gradient clipping is {})"
              .format(optimizer_type, learning_rate, "on" if clip_gradient else "off"))
        self.__optimizer = OptimizerFactory(self.get_parameters(), learning_rate).get(optimizer_type)
        self.__clip_gradient = clip_gradient

    def set_criterion(self, criterion_type: str):
        """
        Instantiates a criterion for the training
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
        print("\n Saving model... \n")
        torch.save(self._network.state_dict(), path_to_model)

    def load(self, path_to_model: str):
        """
        Loads a model from the given path
        :param path_to_model: the path where to load the model from
        """
        print("\n Loading model... \n")
        self._network.load_state_dict(torch.load(path_to_model))

    def _send_network_to_device(self, network: nn.Module):
        self._network = network.float().to(self._device)
